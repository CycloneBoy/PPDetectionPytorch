# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppdettorch.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ppdettorch.loss.detection import GIoULoss, SIoULoss
from ..assigners.utils import generate_anchors_for_grid_cell
from ..backbones.yolov6_efficientrep import BaseConv
from ppdettorch.modeling.ops import get_static_shape, get_act_fn
from ppdettorch.modeling.layers import MultiClassNMS

__all__ = ['EffiDeHead']


@register
class EffiDeHead(nn.Module):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'self_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            in_channels=[128, 256, 512],
            num_classes=80,
            act='relu',
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=False,  # False in n/t/s version, True in m/l version
            static_assigner_epoch=3,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            eval_size=None,
            iou_type='giou',  # 'siou' in n/t version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version 
                'cwd': 10.0,  # used when self_distill=True, in m/l version
            },
            nms='MultiClassNMS',
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            self_distill=False,
            distill_weight={
                'cls': 1.0,
                'dfl': 1.0,
            },
            print_l1_loss=True):
        super(EffiDeHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.eval_size = eval_size
        self.iou_loss = GIoULoss()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # for self-distillation
        self.self_distill = self_distill
        self.distill_weight = distill_weight

        # stem
        self.stem_conv = nn.ModuleList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_conv.append(BaseConv(in_c, in_c, 1, 1))

        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Sequential(*[
                    BaseConv(in_c, in_c, 3, 1), nn.Conv2d(
                        in_c,
                        self.num_classes,
                        1,
                        bias=True)
                ]))
            self.pred_reg.append(
                nn.Sequential(*[
                    BaseConv(in_c, in_c, 3, 1), nn.Conv2d(
                        in_c,
                        4 * (self.reg_max + 1),
                        1,
                        bias=True)
                ]))
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.proj_conv.skip_quant = True
        self.print_l1_loss = print_l1_loss

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            feat = self.stem_conv[i](feat)
            cls_logit = self.pred_cls[i](feat)
            reg_distri = self.pred_reg[i](feat)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute(0, 2, 1))
            reg_distri_list.append(reg_distri.flatten(2).permute(0, 2, 1))
        cls_score_list = torch.concat(cls_score_list, dim=1)
        reg_distri_list = torch.concat(reg_distri_list, dim=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)

    def _generate_anchors(self, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=torch.float32))
        anchor_points = torch.concat(anchor_points)
        stride_tensor = torch.concat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            feat = self.stem_conv[i](feat)
            cls_logit = self.pred_cls[i](feat)
            reg_dist = self.pred_reg[i](feat)
            if self.use_dfl:  ### diff with PPYOLOEHead
                reg_dist = reg_dist.reshape(
                    [-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = torch.concat(cls_score_list, dim=-1)
        reg_dist_list = torch.concat(reg_dist_list, dim=-1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        ### diff with PPYOLOEHead
        if self.use_dfl:
            b, l, _ = get_static_shape(pred_dist)
            pred_dist = F.softmax(
                pred_dist.reshape([b, l, 4, self.reg_max + 1])).matmul(
                self.proj)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.concat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target
        target_right = target_left + 1
        weight_left = target_right - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # l1 loss just see the convergence, same in PPYOLOEHead
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            # dfl loss ### diff with PPYOLOEHead
            if self.use_dfl:
                dist_mask = mask_positive.unsqueeze(-1).tile(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                assigned_ltrb = self._bbox2distance(anchor_points,
                                                    assigned_bboxes)
                assigned_ltrb_pos = torch.masked_select(
                    assigned_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         assigned_ltrb_pos) * bbox_weight
                loss_dfl = loss_dfl.sum() / assigned_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
        else:
            loss_l1 = torch.zeros([1])
            loss_iou = torch.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors, \
            anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor

        # cls loss: varifocal_loss
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                        one_hot_label)
        assigned_scores_sum = assigned_scores.sum()
        if torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(assigned_scores_sum)
        assigned_scores_sum = torch.clip(
            assigned_scores_sum / torch.distributed.get_world_size(),
            min=1)

        loss_cls /= assigned_scores_sum

        # bbox loss
        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        if self.use_dfl:
            loss = self.loss_weight['cls'] * loss_cls + \
                   self.loss_weight['iou'] * loss_iou + \
                   self.loss_weight['dfl'] * loss_dfl
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
                'loss_dfl': loss_dfl,
            }
        else:
            loss = self.loss_weight['cls'] * loss_cls + \
                   self.loss_weight['iou'] * loss_iou
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
            }

        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1})
        return out_dict

    def post_process(self, head_outs, im_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        run_device = pred_scores.device
        anchor_points = anchor_points.to(run_device)
        stride_tensor = stride_tensor.to(run_device)
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist.permute(0, 2, 1))
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        scale_y, scale_x = torch.split(scale_factor, [1, 1], dim=-1)
        scale_factor = torch.concat(
            [scale_x, scale_y, scale_x, scale_y], dim=-1).reshape([-1, 1, 4])
        pred_bboxes /= scale_factor

        if self.exclude_post_process:
            return torch.concat(
                [pred_bboxes, pred_scores.permute(0, 2, 1)],
                dim=-1), torch.IntTensor([1])
        else:
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes.sum(), pred_scores.sum()
            else:
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                return bbox_pred, bbox_num
