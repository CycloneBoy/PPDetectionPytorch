#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PPDetectionPytorch
# @File    ：layers.py
# @Author  ：sl
# @Date    ：2022/11/1 14:25

#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import time
from itertools import chain

import six
import numpy as np
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ppdettorch.modeling.losses.yolo_loss import bbox_transform
from torchvision.ops import DeformConv2d

from ppdettorch.modeling.bbox_utils import delta2bbox, xywh2xyxy_v2, rescale_boxes, yolo_box_pytorch
from ppdettorch.core.workspace import register, serializable

from . import ops


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]


class AlignConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.align_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
            groups=groups)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        """
        Args:
            anchors: [B, L, 5] xc,yc,w,h,angle
            featmap_size: (feat_h, feat_w)
            stride: 8
        Returns:

        """
        batch = anchors.shape[0]
        dtype = anchors.dtype
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype)

        yy, xx = torch.meshgrid(idx, idx)
        xx = torch.reshape(xx, [-1])
        yy = torch.reshape(yy, [-1])

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, dtype=dtype)
        yc = torch.arange(0, feat_h, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)

        xc = torch.reshape(xc, [-1, 1])
        yc = torch.reshape(yc, [-1, 1])
        x_conv = xc + xx
        y_conv = yc + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.split(anchors, 5, dim=-1)
        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w_s / self.kernel_size, h_s / self.kernel_size
        x, y = dw * xx, dh * yy
        xr = cos * x - sin * y
        yr = sin * x + cos * y
        x_anchor, y_anchor = xr + x_ctr, yr + y_ctr
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        offset = torch.stack([offset_y, offset_x], dim=-1)
        offset = offset.reshape(
            [batch, feat_h, feat_w, self.kernel_size * self.kernel_size * 2])
        offset = offset.transpose([0, 3, 1, 2])

        return offset

    def forward(self, x, refine_anchors, featmap_size, stride):
        batch = torch.shape(x)[0].numpy()
        offset = self.get_offset(refine_anchors, featmap_size, stride)
        if self.training:
            x = F.relu(self.align_conv(x, offset.detach()))
        else:
            x = F.relu(self.align_conv(x, offset))
        return x


class DeformableConvV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 lr_scale=1,
                 regularizer=None,
                 skip_quant=False,
                 dcn_bias_regularizer=None,
                 dcn_bias_lr_scale=2.):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size ** 2
        self.mask_channel = kernel_size ** 2

        self.conv_offset = nn.Conv2d(
            in_channels,
            3 * kernel_size ** 2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2)
        if skip_quant:
            self.conv_offset.skip_quant = True

        if bias_attr:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            dcn_bias_attr = False
        else:
            # in ResNet backbone, do not need bias
            dcn_bias_attr = False
        self.conv_dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            bias=dcn_bias_attr)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = torch.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            dim=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 use_dcn=False,
                 bias_on=False,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=None,
                 skip_quant=False,
                 dcn_lr_scale=2.,
                 dcn_regularizer=None):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]

        if not use_dcn:
            self.conv = nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias=bias_on)
            if skip_quant:
                self.conv.skip_quant = True
        else:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias_attr=True,
                # lr_scale=dcn_lr_scale,
                regularizer=dcn_regularizer,
                dcn_bias_regularizer=dcn_regularizer,
                dcn_bias_lr_scale=dcn_lr_scale,
                skip_quant=skip_quant)

        norm_lr = 0. if freeze_norm else 1.

        if norm_type in ['bn', 'sync_bn']:
            self.norm = nn.BatchNorm2d(ch_out)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out)
        else:
            self.norm = None

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm is not None:
            out = self.norm(out)
        return out


class LiteConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 with_act=True,
                 norm_type='sync_bn',
                 name=None):
        super(LiteConv, self).__init__()
        self.lite_conv = nn.Sequential()
        conv1 = ConvNormLayer(
            in_channels,
            in_channels,
            filter_size=5,
            stride=stride,
            groups=in_channels,
            norm_type=norm_type,
        )
        conv2 = ConvNormLayer(
            in_channels,
            out_channels,
            filter_size=1,
            stride=stride,
            norm_type=norm_type,
        )
        conv3 = ConvNormLayer(
            out_channels,
            out_channels,
            filter_size=1,
            stride=stride,
            norm_type=norm_type,
        )
        conv4 = ConvNormLayer(
            out_channels,
            out_channels,
            filter_size=5,
            stride=stride,
            groups=out_channels,
            norm_type=norm_type,
        )
        conv_list = [conv1, conv2, conv3, conv4]
        self.lite_conv.add_sublayer('conv1', conv1)
        self.lite_conv.add_sublayer('relu6_1', nn.ReLU6())
        self.lite_conv.add_sublayer('conv2', conv2)
        if with_act:
            self.lite_conv.add_sublayer('relu6_2', nn.ReLU6())
        self.lite_conv.add_sublayer('conv3', conv3)
        self.lite_conv.add_sublayer('relu6_3', nn.ReLU6())
        self.lite_conv.add_sublayer('conv4', conv4)
        if with_act:
            self.lite_conv.add_sublayer('relu6_4', nn.ReLU6())

    def forward(self, inputs):
        out = self.lite_conv(inputs)
        return out


class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob, name=None, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size ** 2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = torch.Tensor(torch.rand(x.shape) < gamma, x.dtype)
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
                data_format=self.data_format)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


@register
@serializable
class AnchorGeneratorSSD(object):
    def __init__(self,
                 steps=[8, 16, 32, 64, 100, 300],
                 aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
                 min_ratio=15,
                 max_ratio=90,
                 base_size=300,
                 min_sizes=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
                 max_sizes=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
                 offset=0.5,
                 flip=True,
                 clip=False,
                 min_max_aspect_ratios_order=False):
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.base_size = base_size
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.offset = offset
        self.flip = flip
        self.clip = clip
        self.min_max_aspect_ratios_order = min_max_aspect_ratios_order

        if self.min_sizes == [] and self.max_sizes == []:
            num_layer = len(aspect_ratios)
            step = int(
                math.floor(((self.max_ratio - self.min_ratio)) / (num_layer - 2
                                                                  )))
            for ratio in six.moves.range(self.min_ratio, self.max_ratio + 1,
                                         step):
                self.min_sizes.append(self.base_size * ratio / 100.)
                self.max_sizes.append(self.base_size * (ratio + step) / 100.)
            self.min_sizes = [self.base_size * .10] + self.min_sizes
            self.max_sizes = [self.base_size * .20] + self.max_sizes

        self.num_priors = []
        for aspect_ratio, min_size, max_size in zip(
                aspect_ratios, self.min_sizes, self.max_sizes):
            if isinstance(min_size, (list, tuple)):
                self.num_priors.append(
                    len(_to_list(min_size)) + len(_to_list(max_size)))
            else:
                self.num_priors.append((len(aspect_ratio) * 2 + 1) * len(
                    _to_list(min_size)) + len(_to_list(max_size)))

    def __call__(self, inputs, image):
        boxes = []
        for input, min_size, max_size, aspect_ratio, step in zip(
                inputs, self.min_sizes, self.max_sizes, self.aspect_ratios,
                self.steps):
            box, _ = ops.prior_box(
                input=input,
                image=image,
                min_sizes=_to_list(min_size),
                max_sizes=_to_list(max_size),
                aspect_ratios=aspect_ratio,
                flip=self.flip,
                clip=self.clip,
                steps=[step, step],
                offset=self.offset,
                min_max_aspect_ratios_order=self.min_max_aspect_ratios_order)
            boxes.append(torch.reshape(box, [-1, 4]))
        return boxes


@register
@serializable
class RCNNBox(object):
    __shared__ = ['num_classes', 'export_onnx']

    def __init__(self,
                 prior_box_var=[10., 10., 5., 5.],
                 code_type="decode_center_size",
                 box_normalized=False,
                 num_classes=80,
                 export_onnx=False):
        super(RCNNBox, self).__init__()
        self.prior_box_var = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.num_classes = num_classes
        self.export_onnx = export_onnx

    def __call__(self, bbox_head_out, rois, im_shape, scale_factor):
        bbox_pred = bbox_head_out[0]
        cls_prob = bbox_head_out[1]
        roi = rois[0]
        rois_num = rois[1]

        if self.export_onnx:
            onnx_rois_num_per_im = rois_num[0]
            origin_shape = torch.expand(im_shape[0, :],
                                        [onnx_rois_num_per_im, 2])

        else:
            origin_shape_list = []
            if isinstance(roi, list):
                batch_size = len(roi)
            else:
                batch_size = torch.slice(torch.shape(im_shape), [0], [0], [1])

            # bbox_pred.shape: [N, C*4]
            for idx in range(batch_size):
                rois_num_per_im = rois_num[idx]
                expand_im_shape = torch.expand(im_shape[idx, :],
                                               [rois_num_per_im, 2])
                origin_shape_list.append(expand_im_shape)

            origin_shape = torch.concat(origin_shape_list)

        # bbox_pred.shape: [N, C*4]
        # C=num_classes in faster/mask rcnn(bbox_head), C=1 in cascade rcnn(cascade_head)
        bbox = torch.concat(roi)
        bbox = delta2bbox(bbox_pred, bbox, self.prior_box_var)
        scores = cls_prob[:, :-1]

        # bbox.shape: [N, C, 4]
        # bbox.shape[1] must be equal to scores.shape[1]
        total_num = bbox.shape[0]
        bbox_dim = bbox.shape[-1]
        bbox = torch.expand(bbox, [total_num, self.num_classes, bbox_dim])

        origin_h = torch.unsqueeze(origin_shape[:, 0], dim=1)
        origin_w = torch.unsqueeze(origin_shape[:, 1], dim=1)
        zeros = torch.zeros_like(origin_h)
        x1 = torch.maximum(torch.minimum(bbox[:, :, 0], origin_w), zeros)
        y1 = torch.maximum(torch.minimum(bbox[:, :, 1], origin_h), zeros)
        x2 = torch.maximum(torch.minimum(bbox[:, :, 2], origin_w), zeros)
        y2 = torch.maximum(torch.minimum(bbox[:, :, 3], origin_h), zeros)
        bbox = torch.stack([x1, y1, x2, y2], dim=-1)
        bboxes = (bbox, rois_num)
        return bboxes, scores


@register
@serializable
class MultiClassNMS(object):
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=True,
                 nms_eta=1.0,
                 return_index=False,
                 return_rois_num=True,
                 trt=False):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_index = return_index
        self.return_rois_num = return_rois_num
        self.trt = trt

    def __call__(self, bboxes, score, background_label=-1):
        """
        bboxes (Tensor|List[Tensor]): 1. (Tensor) Predicted bboxes with shape
                                         [N, M, 4], N is the batch size and M
                                         is the number of bboxes
                                      2. (List[Tensor]) bboxes and bbox_num,
                                         bboxes have shape of [M, C, 4], C
                                         is the class number and bbox_num means
                                         the number of bboxes of each batch with
                                         shape [N,]
        score (Tensor): Predicted scores with shape [N, C, M] or [M, C]
        background_label (int): Ignore the background label; For example, RCNN
                                is num_classes and YOLO is -1.
        """
        kwargs = self.__dict__.copy()
        if isinstance(bboxes, tuple):
            bboxes, bbox_num = bboxes
            kwargs.update({'rois_num': bbox_num})
        if background_label > -1:
            kwargs.update({'background_label': background_label})
        kwargs.pop('trt')
        # TODO(wangxinxin08): paddle version should be develop or 2.3 and above to run nms on tensorrt
        if self.trt and (int(torch.version.major) == 0 or
                         (int(torch.version.major) >= 2 and
                          int(torch.version.minor) >= 3)):
            # TODO(wangxinxin08): tricky switch to run nms on tensorrt
            kwargs.update({'nms_eta': 1.1})
            bbox, bbox_num, _ = ops.multiclass_nms(bboxes, score, **kwargs)
            bbox = bbox.reshape([1, -1, 6])
            idx = torch.nonzero(bbox[..., 0] != -1)
            bbox = torch.gather_nd(bbox, idx)
            return bbox, bbox_num, None
        else:
            return ops.multiclass_nms(bboxes, score, **kwargs)


@register
@serializable
class MatrixNMS(object):
    __append_doc__ = True

    def __init__(self,
                 score_threshold=.05,
                 post_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 use_gaussian=False,
                 gaussian_sigma=2.,
                 normalized=False,
                 background_label=0):
        super(MatrixNMS, self).__init__()
        self.score_threshold = score_threshold
        self.post_threshold = post_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.normalized = normalized
        self.use_gaussian = use_gaussian
        self.gaussian_sigma = gaussian_sigma
        self.background_label = background_label

    def __call__(self, bbox, score, *args):
        return ops.matrix_nms(
            bboxes=bbox,
            scores=score,
            score_threshold=self.score_threshold,
            post_threshold=self.post_threshold,
            nms_top_k=self.nms_top_k,
            keep_top_k=self.keep_top_k,
            use_gaussian=self.use_gaussian,
            gaussian_sigma=self.gaussian_sigma,
            background_label=self.background_label,
            normalized=self.normalized)


@register
@serializable
class YOLOBox(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 conf_thresh=0.005,
                 downsample_ratio=32,
                 clip_bbox=True,
                 scale_x_y=1.):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.downsample_ratio = downsample_ratio
        self.clip_bbox = clip_bbox
        self.scale_x_y = scale_x_y

        self.sigmoid = nn.Sigmoid()

    def __call__(self,
                 yolo_head_out,
                 anchors,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes_list = []
        scores_list = []
        origin_shape = im_shape / scale_factor
        origin_shape_list = origin_shape.detach().cpu().numpy().tolist()[0]

        # img_size = origin_shape_list[1]
        img_size = im_shape.detach().cpu().numpy().tolist()[0][0]
        # print(f"c {im_shape} -scale_factor: {scale_factor} -origin_shape_list: {origin_shape_list} - img_size: {img_size}")

        for i, head_out in enumerate(yolo_head_out):
            boxes, scores = yolo_box_pytorch(x=head_out,
                                             img_size=origin_shape,
                                             anchors=anchors[i],
                                             class_num=self.num_classes,
                                             conf_thresh=self.conf_thresh,
                                             downsample_ratio=self.downsample_ratio // 2 ** i,
                                             clip_bbox=self.clip_bbox,
                                             scale_x_y=self.scale_x_y, )

            boxes_list.append(boxes)
            scores_list.append(scores.permute(0, 2, 1))

        yolo_boxes = torch.concat(boxes_list, dim=1)
        yolo_scores = torch.concat(scores_list, dim=2)

        return yolo_boxes, yolo_scores


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        print(f"YOLOLayer x : {x.shape}  - {img_size}")
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        print(f"YOLOLayer: {x.shape} - {self.anchor_grid} - {x}")
        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid  # wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(bs, -1, self.no)
        print(f"YOLOLayer output: {x.shape} - {self.anchor_grid} - {x}")
        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


@register
@serializable
class SSDBox(object):
    def __init__(self,
                 is_normalized=True,
                 prior_box_var=[0.1, 0.1, 0.2, 0.2],
                 use_fuse_decode=False):
        self.is_normalized = is_normalized
        self.norm_delta = float(not self.is_normalized)
        self.prior_box_var = prior_box_var
        self.use_fuse_decode = use_fuse_decode

    def __call__(self,
                 preds,
                 prior_boxes,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes, scores = preds
        boxes = torch.concat(boxes, dim=1)
        prior_boxes = torch.concat(prior_boxes)
        if self.use_fuse_decode:
            output_boxes = ops.box_coder(
                prior_boxes,
                self.prior_box_var,
                boxes,
                code_type="decode_center_size",
                box_normalized=self.is_normalized)
        else:
            pb_w = prior_boxes[:, 2] - prior_boxes[:, 0] + self.norm_delta
            pb_h = prior_boxes[:, 3] - prior_boxes[:, 1] + self.norm_delta
            pb_x = prior_boxes[:, 0] + pb_w * 0.5
            pb_y = prior_boxes[:, 1] + pb_h * 0.5
            out_x = pb_x + boxes[:, :, 0] * pb_w * self.prior_box_var[0]
            out_y = pb_y + boxes[:, :, 1] * pb_h * self.prior_box_var[1]
            out_w = torch.exp(boxes[:, :, 2] * self.prior_box_var[2]) * pb_w
            out_h = torch.exp(boxes[:, :, 3] * self.prior_box_var[3]) * pb_h
            output_boxes = torch.stack(
                [
                    out_x - out_w / 2., out_y - out_h / 2., out_x + out_w / 2.,
                    out_y + out_h / 2.
                ],
                dim=-1)

        if self.is_normalized:
            h = (im_shape[:, 0] / scale_factor[:, 0]).unsqueeze(-1)
            w = (im_shape[:, 1] / scale_factor[:, 1]).unsqueeze(-1)
            im_shape = torch.stack([w, h, w, h], dim=-1)
            output_boxes *= im_shape
        else:
            output_boxes[..., -2:] -= 1.0
        output_scores = F.softmax(torch.concat(scores, dim=1), dim=-1).permute(0, 2, 1)

        return output_boxes, output_scores


@register
@serializable
class FCOSBox(object):
    __shared__ = ['num_classes']

    def __init__(self, num_classes=80):
        super(FCOSBox, self).__init__()
        self.num_classes = num_classes

    def _merge_hw(self, inputs, ch_type="channel_first"):
        """
        Merge h and w of the feature map into one dimension.
        Args:
            inputs (Tensor): Tensor of the input feature map
            ch_type (str): "channel_first" or "channel_last" style
        Return:
            new_shape (Tensor): The new shape after h and w merged
        """
        shape_ = torch.shape(inputs)
        bs, ch, hi, wi = shape_[0], shape_[1], shape_[2], shape_[3]
        img_size = hi * wi
        img_size.stop_gradient = True
        if ch_type == "channel_first":
            new_shape = torch.concat([bs, ch, img_size])
        elif ch_type == "channel_last":
            new_shape = torch.concat([bs, img_size, ch])
        else:
            raise KeyError("Wrong ch_type %s" % ch_type)
        new_shape.stop_gradient = True
        return new_shape

    def _postprocessing_by_level(self, locations, box_cls, box_reg, box_ctn,
                                 scale_factor):
        """
        Postprocess each layer of the output with corresponding locations.
        Args:
            locations (Tensor): anchor points for current layer, [H*W, 2]
            box_cls (Tensor): categories prediction, [N, C, H, W],
                C is the number of classes
            box_reg (Tensor): bounding box prediction, [N, 4, H, W]
            box_ctn (Tensor): centerness prediction, [N, 1, H, W]
            scale_factor (Tensor): [h_scale, w_scale] for input images
        Return:
            box_cls_ch_last (Tensor): score for each category, in [N, C, M]
                C is the number of classes and M is the number of anchor points
            box_reg_decoding (Tensor): decoded bounding box, in [N, M, 4]
                last dimension is [x1, y1, x2, y2]
        """
        act_shape_cls = self._merge_hw(box_cls)
        box_cls_ch_last = torch.reshape(x=box_cls, shape=act_shape_cls)
        box_cls_ch_last = F.sigmoid(box_cls_ch_last)

        act_shape_reg = self._merge_hw(box_reg)
        box_reg_ch_last = torch.reshape(x=box_reg, shape=act_shape_reg)
        box_reg_ch_last = torch.transpose(box_reg_ch_last, perm=[0, 2, 1])
        box_reg_decoding = torch.stack(
            [
                locations[:, 0] - box_reg_ch_last[:, :, 0],
                locations[:, 1] - box_reg_ch_last[:, :, 1],
                locations[:, 0] + box_reg_ch_last[:, :, 2],
                locations[:, 1] + box_reg_ch_last[:, :, 3]
            ],
            dim=1)
        box_reg_decoding = torch.transpose(box_reg_decoding, perm=[0, 2, 1])

        act_shape_ctn = self._merge_hw(box_ctn)
        box_ctn_ch_last = torch.reshape(x=box_ctn, shape=act_shape_ctn)
        box_ctn_ch_last = F.sigmoid(box_ctn_ch_last)

        # recover the location to original image
        im_scale = torch.concat([scale_factor, scale_factor], dim=1)
        im_scale = torch.expand(im_scale, [box_reg_decoding.shape[0], 4])
        im_scale = torch.reshape(im_scale, [box_reg_decoding.shape[0], -1, 4])
        box_reg_decoding = box_reg_decoding / im_scale
        box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last
        return box_cls_ch_last, box_reg_decoding

    def __call__(self, locations, cls_logits, bboxes_reg, centerness,
                 scale_factor):
        pred_boxes_ = []
        pred_scores_ = []
        for pts, cls, box, ctn in zip(locations, cls_logits, bboxes_reg,
                                      centerness):
            pred_scores_lvl, pred_boxes_lvl = self._postprocessing_by_level(
                pts, cls, box, ctn, scale_factor)
            pred_boxes_.append(pred_boxes_lvl)
            pred_scores_.append(pred_scores_lvl)
        pred_boxes = torch.concat(pred_boxes_, dim=1)
        pred_scores = torch.concat(pred_scores_, dim=2)
        return pred_boxes, pred_scores


@register
class TTFBox(object):
    __shared__ = ['down_ratio']

    def __init__(self, max_per_img=100, score_thresh=0.01, down_ratio=4):
        super(TTFBox, self).__init__()
        self.max_per_img = max_per_img
        self.score_thresh = score_thresh
        self.down_ratio = down_ratio

    def _simple_nms(self, heat, kernel=3):
        """
        Use maxpool to filter the max score, get local peaks.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = torch.FloatTensor(hmax == heat)
        return heat * keep

    def _topk(self, scores):
        """
        Select top k scores and decode to get xy coordinates.
        """
        k = self.max_per_img
        shape_fm = scores.shape
        shape_fm.stop_gradient = True
        cat, height, width = shape_fm[1], shape_fm[2], shape_fm[3]
        # batch size is 1
        scores_r = torch.reshape(scores, [cat, -1])
        topk_scores, topk_inds = torch.topk(scores_r, k)
        topk_ys = topk_inds // width
        topk_xs = topk_inds % width

        topk_score_r = torch.reshape(topk_scores, [-1])
        topk_score, topk_ind = torch.topk(topk_score_r, k)
        k_t = torch.full(topk_ind.shape, k, dtype=torch.long)
        topk_clses = torch.FloatTensor(torch.floor_divide(topk_ind, k_t))

        topk_inds = torch.reshape(topk_inds, [-1])
        topk_ys = torch.reshape(topk_ys, [-1, 1])
        topk_xs = torch.reshape(topk_xs, [-1, 1])
        topk_inds = torch.gather(topk_inds, topk_ind)
        topk_ys = torch.gather(topk_ys, topk_ind)
        topk_xs = torch.gather(topk_xs, topk_ind)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _decode(self, hm, wh, im_shape, scale_factor):
        heatmap = F.sigmoid(hm)
        heat = self._simple_nms(heatmap)
        scores, inds, clses, ys, xs = self._topk(heat)
        ys = torch.cast(ys, 'float32') * self.down_ratio
        xs = torch.cast(xs, 'float32') * self.down_ratio
        scores = torch.tensor.unsqueeze(scores, [1])
        clses = torch.tensor.unsqueeze(clses, [1])

        wh_t = torch.transpose(wh, [0, 2, 3, 1])
        wh = torch.reshape(wh_t, [-1, torch.shape(wh_t)[-1]])
        wh = torch.gather(wh, inds)

        x1 = xs - wh[:, 0:1]
        y1 = ys - wh[:, 1:2]
        x2 = xs + wh[:, 2:3]
        y2 = ys + wh[:, 3:4]

        bboxes = torch.concat([x1, y1, x2, y2], dim=1)

        scale_y = scale_factor[:, 0:1]
        scale_x = scale_factor[:, 1:2]
        scale_expand = torch.concat(
            [scale_x, scale_y, scale_x, scale_y], dim=1)
        boxes_shape = torch.shape(bboxes)
        boxes_shape.stop_gradient = True
        scale_expand = torch.expand(scale_expand, shape=boxes_shape)
        bboxes = torch.divide(bboxes, scale_expand)
        results = torch.concat([clses, scores, bboxes], dim=1)
        # hack: append result with cls=-1 and score=1. to avoid all scores
        # are less than score_thresh which may cause error in gather.
        fill_r = torch.to_tensor(np.array([[-1, 1, 0, 0, 0, 0]]))
        fill_r = torch.cast(fill_r, results.dtype)
        results = torch.concat([results, fill_r])
        scores = results[:, 1]
        valid_ind = torch.nonzero(scores > self.score_thresh)
        results = torch.gather(results, valid_ind)
        return results, torch.shape(results)[0:1]

    def __call__(self, hm, wh, im_shape, scale_factor):
        results = []
        results_num = []
        for i in range(scale_factor.shape[0]):
            result, num = self._decode(hm[i:i + 1, ], wh[i:i + 1, ],
                                       im_shape[i:i + 1, ],
                                       scale_factor[i:i + 1, ])
            results.append(result)
            results_num.append(num)
        results = torch.concat(results, dim=0)
        results_num = torch.concat(results_num, dim=0)
        return results, results_num


@register
@serializable
class JDEBox(object):
    __shared__ = ['num_classes']

    def __init__(self, num_classes=1, conf_thresh=0.3, downsample_ratio=32):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.downsample_ratio = downsample_ratio

    def generate_anchor(self, nGh, nGw, anchor_wh):
        nA = len(anchor_wh)
        yv, xv = torch.meshgrid([torch.arange(nGh), torch.arange(nGw)])
        mesh = torch.stack(
            (xv, yv), dim=0).cast(dtype='float32')  # 2 x nGh x nGw
        meshs = torch.tile(mesh, [nA, 1, 1, 1])

        anchor_offset_mesh = anchor_wh[:, :, None][:, :, :, None].repeat(
            int(nGh), dim=-2).repeat(
            int(nGw), dim=-1)
        anchor_offset_mesh = torch.to_tensor(
            anchor_offset_mesh.astype(np.float32))
        # nA x 2 x nGh x nGw

        anchor_mesh = torch.concat([meshs, anchor_offset_mesh], dim=1)
        anchor_mesh = torch.transpose(anchor_mesh,
                                      [0, 2, 3, 1])  # (nA x nGh x nGw) x 4
        return anchor_mesh

    def decode_delta(self, delta, fg_anchor_list):
        px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:, 1], \
            fg_anchor_list[:, 2], fg_anchor_list[:, 3]
        dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
        gx = pw * dx + px
        gy = ph * dy + py
        gw = pw * torch.exp(dw)
        gh = ph * torch.exp(dh)
        gx1 = gx - gw * 0.5
        gy1 = gy - gh * 0.5
        gx2 = gx + gw * 0.5
        gy2 = gy + gh * 0.5
        return torch.stack([gx1, gy1, gx2, gy2], dim=1)

    def decode_delta_map(self, nA, nGh, nGw, delta_map, anchor_vec):
        anchor_mesh = self.generate_anchor(nGh, nGw, anchor_vec)
        anchor_mesh = torch.unsqueeze(anchor_mesh, 0)
        pred_list = self.decode_delta(
            torch.reshape(
                delta_map, shape=[-1, 4]),
            torch.reshape(
                anchor_mesh, shape=[-1, 4]))
        pred_map = torch.reshape(pred_list, shape=[nA * nGh * nGw, 4])
        return pred_map

    def _postprocessing_by_level(self, nA, stride, head_out, anchor_vec):
        boxes_shape = head_out.shape  # [nB, nA*6, nGh, nGw]
        nGh, nGw = boxes_shape[-2], boxes_shape[-1]
        nB = 1  # TODO: only support bs=1 now
        boxes_list, scores_list = [], []
        for idx in range(nB):
            p = torch.reshape(
                head_out[idx], shape=[nA, self.num_classes + 5, nGh, nGw])
            p = torch.transpose(p, perm=[0, 2, 3, 1])  # [nA, nGh, nGw, 6]
            delta_map = p[:, :, :, :4]
            boxes = self.decode_delta_map(nA, nGh, nGw, delta_map, anchor_vec)
            # [nA * nGh * nGw, 4]
            boxes_list.append(boxes * stride)

            p_conf = torch.transpose(
                p[:, :, :, 4:6], perm=[3, 0, 1, 2])  # [2, nA, nGh, nGw]
            p_conf = F.softmax(
                p_conf, dim=0)[1, :, :, :].unsqueeze(-1)  # [nA, nGh, nGw, 1]
            scores = torch.reshape(p_conf, shape=[nA * nGh * nGw, 1])
            scores_list.append(scores)

        boxes_results = torch.stack(boxes_list)
        scores_results = torch.stack(scores_list)
        return boxes_results, scores_results

    def __call__(self, yolo_head_out, anchors):
        bbox_pred_list = []
        for i, head_out in enumerate(yolo_head_out):
            stride = self.downsample_ratio // 2 ** i
            anc_w, anc_h = anchors[i][0::2], anchors[i][1::2]
            anchor_vec = np.stack((anc_w, anc_h), dim=1) / stride
            nA = len(anc_w)
            boxes, scores = self._postprocessing_by_level(nA, stride, head_out,
                                                          anchor_vec)
            bbox_pred_list.append(torch.concat([boxes, scores], dim=-1))

        yolo_boxes_scores = torch.concat(bbox_pred_list, dim=1)
        boxes_idx_over_conf_thr = torch.nonzero(
            yolo_boxes_scores[:, :, -1] > self.conf_thresh)
        boxes_idx_over_conf_thr.stop_gradient = True

        return boxes_idx_over_conf_thr, yolo_boxes_scores


@register
@serializable
class MaskMatrixNMS(object):
    """
    Matrix NMS for multi-class masks.
    Args:
        update_threshold (float): Updated threshold of categroy score in second time.
        pre_nms_top_n (int): Number of total instance to be kept per image before NMS
        post_nms_top_n (int): Number of total instance to be kept per image after NMS.
        kernel (str):  'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
    Input:
        seg_preds (Variable): shape (n, h, w), segmentation feature maps
        seg_masks (Variable): shape (n, h, w), segmentation feature maps
        cate_labels (Variable): shape (n), mask labels in descending order
        cate_scores (Variable): shape (n), mask scores in descending order
        sum_masks (Variable): a float tensor of the sum of seg_masks
    Returns:
        Variable: cate_scores, tensors of shape (n)
    """

    def __init__(self,
                 update_threshold=0.05,
                 pre_nms_top_n=500,
                 post_nms_top_n=100,
                 kernel='gaussian',
                 sigma=2.0):
        super(MaskMatrixNMS, self).__init__()
        self.update_threshold = update_threshold
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.kernel = kernel
        self.sigma = sigma

    def _sort_score(self, scores, top_num):
        if torch.shape(scores)[0] > top_num:
            return torch.topk(scores, top_num)[1]
        else:
            return torch.argsort(scores, descending=True)

    def __call__(self,
                 seg_preds,
                 seg_masks,
                 cate_labels,
                 cate_scores,
                 sum_masks=None):
        # sort and keep top nms_pre
        sort_inds = self._sort_score(cate_scores, self.pre_nms_top_n)
        seg_masks = torch.gather(seg_masks, index=sort_inds)
        seg_preds = torch.gather(seg_preds, index=sort_inds)
        sum_masks = torch.gather(sum_masks, index=sort_inds)
        cate_scores = torch.gather(cate_scores, index=sort_inds)
        cate_labels = torch.gather(cate_labels, index=sort_inds)

        seg_masks = torch.flatten(seg_masks, start_dim=1, stop_dim=-1)
        # inter.
        inter_matrix = torch.mm(seg_masks, torch.transpose(seg_masks, [1, 0]))
        n_samples = torch.shape(cate_labels)
        # union.
        sum_masks_x = torch.expand(sum_masks, shape=[n_samples, n_samples])
        # iou.
        iou_matrix = (inter_matrix / (
                sum_masks_x + torch.transpose(sum_masks_x, [1, 0]) - inter_matrix))
        iou_matrix = torch.triu(iou_matrix, diagonal=1)
        # label_specific matrix.
        cate_labels_x = torch.expand(cate_labels, shape=[n_samples, n_samples])
        label_matrix = torch.cast(
            (cate_labels_x == torch.transpose(cate_labels_x, [1, 0])),
            'float32')
        label_matrix = torch.triu(label_matrix, diagonal=1)

        # IoU compensation
        compensate_iou = torch.max((iou_matrix * label_matrix), dim=0)
        compensate_iou = torch.expand(
            compensate_iou, shape=[n_samples, n_samples])
        compensate_iou = torch.transpose(compensate_iou, [1, 0])

        # IoU decay
        decay_iou = iou_matrix * label_matrix

        # matrix nms
        if self.kernel == 'gaussian':
            decay_matrix = torch.exp(-1 * self.sigma * (decay_iou ** 2))
            compensate_matrix = torch.exp(-1 * self.sigma *
                                          (compensate_iou ** 2))
            decay_coefficient = torch.min(decay_matrix / compensate_matrix,
                                          dim=0)
        elif self.kernel == 'linear':
            decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
            decay_coefficient = torch.min(decay_matrix, dim=0)
        else:
            raise NotImplementedError

        # update the score.
        cate_scores = cate_scores * decay_coefficient
        y = torch.zeros(shape=torch.shape(cate_scores), dtype='float32')
        keep = torch.where(cate_scores >= self.update_threshold, cate_scores,
                           y)
        keep = torch.nonzero(keep)
        keep = torch.squeeze(keep, dim=[1])
        # Prevent empty and increase fake data
        keep = torch.concat(
            [keep, torch.cast(torch.shape(cate_scores)[0] - 1, 'int64')])

        seg_preds = torch.gather(seg_preds, index=keep)
        cate_scores = torch.gather(cate_scores, index=keep)
        cate_labels = torch.gather(cate_labels, index=keep)

        # sort and keep top_k
        sort_inds = self._sort_score(cate_scores, self.post_nms_top_n)
        seg_preds = torch.gather(seg_preds, index=sort_inds)
        cate_scores = torch.gather(cate_scores, index=sort_inds)
        cate_labels = torch.gather(cate_labels, index=sort_inds)
        return seg_preds, cate_scores, cate_labels


def Conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           bias=True):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups)
    return conv


def ConvTranspose2d(in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    groups=1,
                    bias=True,
                    dilation=1, ):
    conv = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        bias,
        dilation, )
    return conv


def BatchNorm2d(num_features, eps=1e-05, momentum=0.9, affine=True):
    batchnorm = nn.BatchNorm2d(
        num_features,
        momentum,
        eps,
        affine=affine)
    return batchnorm


def ReLU():
    return nn.ReLU()


def Upsample(scale_factor=None, mode='nearest', align_corners=False):
    return nn.Upsample(None, scale_factor, mode, align_corners)


def MaxPool(kernel_size, stride, padding, ceil_mode=False):
    return nn.MaxPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode)


class Concat(nn.Module):
    def __init__(self, dim=0):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.concat(inputs, dim=self.dim)

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.
    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.
    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    return nn.Module.transformer._convert_attention_mask(attn_mask, dtype)


class MultiHeadAttention(nn.Module):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.

    Examples:

        .. code-block:: python

            import torch

            # encoder input: [batch_size, sequence_length, d_model]
            query = torch.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = torch.rand((2, 2, 4, 4))
            multi_head_attn = torch.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=False)
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=True)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ('q_proj', 'k_proj', 'v_proj')

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                constant_(p)

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            tensor = F.linear(
                x=tensor,
                weight=self.in_proj_weight[:, index * self.embed_dim:(index + 1)
                                                                     * self.embed_dim],
                bias=self.in_proj_bias[index * self.embed_dim:(index + 1) *
                                                              self.embed_dim]
                if self.in_proj_bias is not None else None)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        tensor = tensor.reshape(
            [0, 0, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i)
                   for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = torch.matmul(x=q, y=k, transpose_y=True)
        scaling = float(self.head_dim) ** -0.5
        product = product * scaling

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = torch.matmul(weights, v)

        # combine heads
        out = torch.transpose(out, perm=[0, 2, 1, 3])
        out = torch.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


@register
class ConvMixer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            kernel_size=3, ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size

        self.mixer = self.conv_mixer(dim, depth, kernel_size)

    def forward(self, x):
        return self.mixer(x)

    @staticmethod
    def conv_mixer(
            dim,
            depth,
            kernel_size, ):
        Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(dim))
        Residual = type('Residual', (Seq,),
                        {'forward': lambda self, x: self[0](x) + x})
        return Seq(*[
            Seq(Residual(
                ActBn(
                    nn.Conv2d(
                        dim, dim, kernel_size, groups=dim, padding="same"))),
                ActBn(nn.Conv2d(dim, dim, 1))) for i in range(depth)
        ])
