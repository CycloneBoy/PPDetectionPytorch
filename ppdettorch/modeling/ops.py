# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from torch import Tensor

from ppdettorch.modeling.bbox_utils import ssd_prior_box_np

__all__ = [
    'prior_box',
    'generate_proposals',
    'box_coder',
    'multiclass_nms',
    'distribute_fpn_proposals',
    'matrix_nms',
    'batch_norm',
    'mish',
    'silu',
    'swish',
    'identity',
]


def identity(x):
    return x


def mish(x):
    return F.mish(x) if hasattr(F, mish) else x * F.tanh(F.softplus(x))


def silu(x):
    return F.silu(x)


def swish(x):
    return x * F.sigmoid(x)


TRT_ACT_SPEC = {'swish': swish, 'silu': swish}

ACT_SPEC = {'mish': mish, 'silu': silu, 'swish': silu, }


def get_act_fn(act=None, trt=False):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'
    if not act:
        return identity

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    if trt and name in TRT_ACT_SPEC:
        fn = TRT_ACT_SPEC[name]
    elif name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)


def batch_norm(ch,
               norm_type='bn',
               norm_decay=0.,
               freeze_norm=False,
               initializer=None,
               data_format='NCHW'):
    norm_lr = 0. if freeze_norm else 1.
    weight_attr = ParamAttr(
        initializer=initializer,
        learning_rate=norm_lr,
        regularizer=L2Decay(norm_decay),
        trainable=False if freeze_norm else True)
    bias_attr = ParamAttr(
        learning_rate=norm_lr,
        regularizer=L2Decay(norm_decay),
        trainable=False if freeze_norm else True)

    if norm_type in ['sync_bn', 'bn']:
        norm_layer = nn.BatchNorm2d(
            ch,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    norm_params = norm_layer.parameters()
    if freeze_norm:
        for param in norm_params:
            param.stop_gradient = True

    return norm_layer


# @torch.jit.not_to_static
def distribute_fpn_proposals(fpn_rois,
                             min_level,
                             max_level,
                             refer_level,
                             refer_scale,
                             pixel_offset=False,
                             rois_num=None,
                             name=None):
    r"""

    **This op only takes LoDTensor as input.** In Feature Pyramid Networks 
    (FPN) models, it is needed to distribute all proposals into different FPN 
    level, with respect to scale of the proposals, the referring scale and the 
    referring level. Besides, to restore the order of proposals, we return an 
    array which indicates the original index of rois in current proposals. 
    To compute FPN level for each roi, the formula is given as follows:

    .. math::

        roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}

        level = floor(&\log(\\frac{roi\_scale}{refer\_scale}) + refer\_level)

    where BBoxArea is a function to compute the area of each roi.

    Args:

        fpn_rois(Variable): 2-D Tensor with shape [N, 4] and data type is 
            float32 or float64. The input fpn_rois.
        min_level(int32): The lowest level of FPN layer where the proposals come 
            from.
        max_level(int32): The highest level of FPN layer where the proposals
            come from.
        refer_level(int32): The referring level of FPN layer with specified scale.
        refer_scale(int32): The referring scale of FPN layer with specified level.
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Tuple:

        multi_rois(List) : A list of 2-D LoDTensor with shape [M, 4] 
        and data type of float32 and float64. The length is 
        max_level-min_level+1. The proposals in each FPN level.

        restore_ind(Variable): A 2-D Tensor with shape [N, 1], N is 
        the number of total rois. The data type is int32. It is
        used to restore the order of fpn_rois.

        rois_num_per_level(List): A list of 1-D Tensor and each Tensor is 
        the RoIs' number in each image on the corresponding level. The shape 
        is [B] and data type of int32. B is the number of images


    Examples:
        .. code-block:: python

            import torch
            from ppdettorch.modeling import ops
            torch.enable_static()
            fpn_rois = torch.static.data(
                name='data', shape=[None, 4], dtype='float32', lod_level=1)
            multi_rois, restore_ind = ops.distribute_fpn_proposals(
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224)
    """
    num_lvl = max_level - min_level + 1

    if in_dynamic_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        attrs = ('min_level', min_level, 'max_level', max_level, 'refer_level',
                 refer_level, 'refer_scale', refer_scale, 'pixel_offset',
                 pixel_offset)
        multi_rois, restore_ind, rois_num_per_level = C_ops.distribute_fpn_proposals(
            fpn_rois, rois_num, num_lvl, num_lvl, *attrs)

        return multi_rois, restore_ind, rois_num_per_level

    else:
        check_variable_and_dtype(fpn_rois, 'fpn_rois', ['float32', 'float64'],
                                 'distribute_fpn_proposals')
        helper = LayerHelper('distribute_fpn_proposals', **locals())
        dtype = helper.input_dtype('fpn_rois')
        multi_rois = [
            helper.create_variable_for_type_inference(dtype)
            for i in range(num_lvl)
        ]

        restore_ind = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {'FpnRois': fpn_rois}
        outputs = {
            'MultiFpnRois': multi_rois,
            'RestoreIndex': restore_ind,
        }

        if rois_num is not None:
            inputs['RoisNum'] = rois_num
            rois_num_per_level = [
                helper.create_variable_for_type_inference(dtype='int32')
                for i in range(num_lvl)
            ]
            outputs['MultiLevelRoIsNum'] = rois_num_per_level
        else:
            rois_num_per_level = None

        helper.append_op(
            type='distribute_fpn_proposals',
            inputs=inputs,
            outputs=outputs,
            attrs={
                'min_level': min_level,
                'max_level': max_level,
                'refer_level': refer_level,
                'refer_scale': refer_scale,
                'pixel_offset': pixel_offset
            })
        return multi_rois, restore_ind, rois_num_per_level


# @torch.jit.not_to_static
def prior_box(input,
              image,
              min_sizes,
              max_sizes=None,
              aspect_ratios=[1.],
              variance=[0.1, 0.1, 0.2, 0.2],
              flip=False,
              clip=False,
              steps=[0.0, 0.0],
              offset=0.5,
              min_max_aspect_ratios_order=False,
              name=None):
    """

    This op generates prior boxes for SSD(Single Shot MultiBox Detector) algorithm.
    Each position of the input produce N prior boxes, N is determined by
    the count of min_sizes, max_sizes and aspect_ratios, The size of the
    box is in range(min_size, max_size) interval, which is generated in
    sequence according to the aspect_ratios.

    Parameters:
       input(Tensor): 4-D tensor(NCHW), the data type should be float32 or float64.
       image(Tensor): 4-D tensor(NCHW), the input image data of PriorBoxOp,
            the data type should be float32 or float64.
       min_sizes(list|tuple|float): the min sizes of generated prior boxes.
       max_sizes(list|tuple|None): the max sizes of generated prior boxes.
            Default: None.
       aspect_ratios(list|tuple|float): the aspect ratios of generated
            prior boxes. Default: [1.].
       variance(list|tuple): the variances to be encoded in prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       flip(bool): Whether to flip aspect ratios. Default:False.
       clip(bool): Whether to clip out-of-boundary boxes. Default: False.
       step(list|tuple): Prior boxes step across width and height, If
            step[0] equals to 0.0 or step[1] equals to 0.0, the prior boxes step across
            height or weight of the input will be automatically calculated.
            Default: [0., 0.]
       offset(float): Prior boxes center offset. Default: 0.5
       min_max_aspect_ratios_order(bool): If set True, the output prior box is
            in order of [min, max, aspect_ratios], which is consistent with
            Caffe. Please note, this order affects the weights order of
            convolution layer followed by and does not affect the final
            detection results. Default: False.
       name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tuple: A tuple with two Variable (boxes, variances)

        boxes(Tensor): the output prior boxes of PriorBox.
        4-D tensor, the layout is [H, W, num_priors, 4].
        H is the height of input, W is the width of input,
        num_priors is the total box count of each position of input.

        variances(Tensor): the expanded variances of PriorBox.
        4-D tensor, the layput is [H, W, num_priors, 4].
        H is the height of input, W is the width of input
        num_priors is the total box count of each position of input

    Examples:
        .. code-block:: python

        import torch
        from finanicial_ner.model.detection.modeling import ops

        torch.enable_static()
        input = torch.static.data(name="input", shape=[None,3,6,9])
        image = torch.static.data(name="image", shape=[None,3,9,12])
        box, var = ops.prior_box(
                    input=input,
                    image=image,
                    min_sizes=[100.],
                    clip=True,
                    flip=True)
    """

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if not _is_list_or_tuple_(min_sizes):
        min_sizes = [min_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not (_is_list_or_tuple_(steps) and len(steps) == 2):
        raise ValueError('steps should be a list or tuple ',
                         'with length 2, (step_width, step_height).')

    min_sizes = list(map(float, min_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    steps = list(map(float, steps))

    cur_max_sizes = None
    if max_sizes is not None and len(max_sizes) > 0 and max_sizes[0] > 0:
        if not _is_list_or_tuple_(max_sizes):
            max_sizes = [max_sizes]
        cur_max_sizes = max_sizes

    attrs = ('min_sizes', min_sizes, 'aspect_ratios', aspect_ratios,
             'variances', variance, 'flip', flip, 'clip', clip, 'step_w',
             steps[0], 'step_h', steps[1], 'offset', offset,
             'min_max_aspect_ratios_order', min_max_aspect_ratios_order)
    if cur_max_sizes is not None:
        attrs += ('max_sizes', cur_max_sizes)
    # box, var = C_ops.prior_box(input, image, *attrs)
    box, var = ssd_prior_box_np(input,
                                image,
                                min_sizes=min_sizes,
                                aspect_ratios=aspect_ratios,
                                variances=variance,
                                max_sizes=max_sizes,
                                flip=flip,
                                clip=clip,
                                step_w=steps[0],
                                step_h=steps[1],
                                offset=offset,
                                min_max_aspect_ratios_order=min_max_aspect_ratios_order, )
    run_device = input.device
    box_ret = torch.tensor(box, device=run_device)
    var_ret = torch.tensor(var, device=run_device)
    return box_ret, var_ret


def batched_nms_mmdet(boxes: Tensor,
                      scores: Tensor,
                      idxs: Tensor,
                      nms_cfg: Optional[Dict],
                      class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (
                    max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (
                    max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def multiclass_nms_mmdet(multi_bboxes,
                         multi_scores,
                         score_thr,
                         nms_cfg,
                         max_num=-1,
                         score_factors=None,
                         return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms_mmdet(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    nms_cfg_.pop("type", "nms")
    split_thr = nms_cfg_.pop("split_thr", 10000)
    if len(boxes_for_nms) < split_thr:
        keep = nms(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


def multiclass_nms(bboxes,
                   scores,
                   score_threshold,
                   nms_top_k=-1,
                   keep_top_k=-1,
                   nms_threshold=0.3,
                   normalized=True,
                   nms_eta=1.,
                   background_label=-1,
                   return_index=False,
                   return_rois_num=True,
                   rois_num=None,
                   name=None):
    nms_cfg = {
        "iou_threshold": nms_threshold
    }
    dets, labels = multiclass_nms_v2(multi_bboxes=bboxes,
                                     multi_scores=scores,
                                     score_thr=score_threshold,
                                     nms_cfg=nms_cfg, max_num=keep_top_k, score_factors=None)

    # dets, labels, index = multiclass_nms_mmdet(multi_bboxes=bboxes,
    #                                            multi_scores=scores,
    #                                            score_thr=score_threshold,
    #                                            nms_cfg=nms_cfg,
    #                                            max_num=keep_top_k,
    #                                            score_factors=None,
    #                                            return_inds=True)

    index = None
    nms_rois_num = torch.IntTensor([len(labels)]).to(dets.device)
    labels = labels.unsqueeze(1)
    det_bboxs, det_scores = dets.split([4, 1], dim=1)
    output = torch.cat([labels, det_scores, det_bboxs], dim=-1)
    if rois_num is not None:
        nms_rois_num = rois_num
    if not return_index:
        index = None
    return output, nms_rois_num, index


def multiclass_nms_v2(
        multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


# @torch.jit.not_to_static
def multiclass_nms_default(bboxes,
                           scores,
                           score_threshold,
                           nms_top_k,
                           keep_top_k,
                           nms_threshold=0.3,
                           normalized=True,
                           nms_eta=1.,
                           background_label=-1,
                           return_index=False,
                           return_rois_num=True,
                           rois_num=None,
                           name=None):
    """
    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.
    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): Two types of bboxes are supported:
                           1. (Tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the
                           class number
        scores (Tensor): Two types of scores are supported:
                           1. (Tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str): Name of the multiclass nms op. Default: None.
    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.
        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.
        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
        or A 2-D LoDTensor with shape [No, 10] represents the detections.
        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,
        x4, y4]. No is the total number of detections.
        If all images have not detected results, all elements in LoD will be
        0, and output tensor is empty (None).
        Index: Only return when return_index is True. A 2-D LoDTensor with
        shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.
    Examples:
        .. code-block:: python

            import torch
            from ppdettorch.modeling import ops
            boxes = torch.static.data(name='bboxes', shape=[81, 4],
                                      dtype='float32', lod_level=1)
            scores = torch.static.data(name='scores', shape=[81],
                                      dtype='float32', lod_level=1)
            out, index = ops.multiclass_nms(bboxes=boxes,
                                            scores=scores,
                                            background_label=0,
                                            score_threshold=0.5,
                                            nms_top_k=400,
                                            nms_threshold=0.3,
                                            keep_top_k=200,
                                            normalized=False,
                                            return_index=True)
    """
    helper = LayerHelper('multiclass_nms3', **locals())

    if in_dynamic_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold',
                 nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta,
                 'normalized', normalized)
        output, index, nms_rois_num = C_ops.multiclass_nms3(bboxes, scores,
                                                            rois_num, *attrs)
        if not return_index:
            index = None
        return output, nms_rois_num, index

    else:
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {'BBoxes': bboxes, 'Scores': scores}
        outputs = {'Out': output, 'Index': index}

        if rois_num is not None:
            inputs['RoisNum'] = rois_num

        if return_rois_num:
            nms_rois_num = helper.create_variable_for_type_inference(
                dtype='int32')
            outputs['NmsRoisNum'] = nms_rois_num

        helper.append_op(
            type="multiclass_nms3",
            inputs=inputs,
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'nms_top_k': nms_top_k,
                'nms_threshold': nms_threshold,
                'keep_top_k': keep_top_k,
                'nms_eta': nms_eta,
                'normalized': normalized
            },
            outputs=outputs)
        output.stop_gradient = True
        index.stop_gradient = True
        if not return_index:
            index = None
        if not return_rois_num:
            nms_rois_num = None

        return output, nms_rois_num, index


# @torch.jit.not_to_static
def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.,
               background_label=0,
               normalized=True,
               return_index=False,
               return_rois_num=True,
               name=None):
    """
    **Matrix NMS**
    This operator does matrix non maximum suppression (NMS).
    First selects a subset of candidate bounding boxes that have higher scores
    than score_threshold (if provided), then the top k candidate is selected if
    nms_top_k is larger than -1. Score of the remaining candidate are then
    decayed according to the Matrix NMS scheme.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): A 3-D Tensor with shape [N, M, 4] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           The data type is float32 or float64.
        scores (Tensor): A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes. The data type is float32 or float64.
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score.
        post_threshold (float): Threshold to filter out bounding boxes with
                                low confidence score AFTER decaying.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        use_gaussian (bool): Use Gaussian as the decay function. Default: False
        gaussian_sigma (float): Sigma for Gaussian decay function. Default: 2.0
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        return_rois_num(bool): whether return rois_num. Default: True
        name(str): Name of the matrix nms op. Default: None.
    Returns:
        A tuple with three Tensor: (Out, Index, RoisNum) if return_index is True,
        otherwise, a tuple with two Tensor (Out, RoisNum) is returned.
        Out (Tensor): A 2-D Tensor with shape [No, 6] containing the
             detection results.
             Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
             (After version 1.3, when no boxes detected, the lod is changed
             from {0} to {1})
        Index (Tensor): A 2-D Tensor with shape [No, 1] containing the
            selected indices, which are absolute values cross batches.
        rois_num (Tensor): A 1-D Tensor with shape [N] containing 
            the number of detected boxes in each image.
    Examples:
        .. code-block:: python
            import torch
            from ppdettorch.modeling import ops
            boxes = torch.static.data(name='bboxes', shape=[None,81, 4],
                                      dtype='float32', lod_level=1)
            scores = torch.static.data(name='scores', shape=[None,81],
                                      dtype='float32', lod_level=1)
            out = ops.matrix_nms(bboxes=boxes, scores=scores, background_label=0,
                                 score_threshold=0.5, post_threshold=0.1,
                                 nms_top_k=400, keep_top_k=200, normalized=False)
    """
    check_variable_and_dtype(bboxes, 'BBoxes', ['float32', 'float64'],
                             'matrix_nms')
    check_variable_and_dtype(scores, 'Scores', ['float32', 'float64'],
                             'matrix_nms')
    check_type(score_threshold, 'score_threshold', float, 'matrix_nms')
    check_type(post_threshold, 'post_threshold', float, 'matrix_nms')
    check_type(nms_top_k, 'nums_top_k', int, 'matrix_nms')
    check_type(keep_top_k, 'keep_top_k', int, 'matrix_nms')
    check_type(normalized, 'normalized', bool, 'matrix_nms')
    check_type(use_gaussian, 'use_gaussian', bool, 'matrix_nms')
    check_type(gaussian_sigma, 'gaussian_sigma', float, 'matrix_nms')
    check_type(background_label, 'background_label', int, 'matrix_nms')

    if in_dynamic_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'post_threshold', post_threshold, 'nms_top_k',
                 nms_top_k, 'gaussian_sigma', gaussian_sigma, 'use_gaussian',
                 use_gaussian, 'keep_top_k', keep_top_k, 'normalized',
                 normalized)
        out, index, rois_num = C_ops.matrix_nms(bboxes, scores, *attrs)
        if not return_index:
            index = None
        if not return_rois_num:
            rois_num = None
        return out, rois_num, index
    else:
        helper = LayerHelper('matrix_nms', **locals())
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int32')
        outputs = {'Out': output, 'Index': index}
        if return_rois_num:
            rois_num = helper.create_variable_for_type_inference(dtype='int32')
            outputs['RoisNum'] = rois_num

        helper.append_op(
            type="matrix_nms",
            inputs={'BBoxes': bboxes,
                    'Scores': scores},
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'post_threshold': post_threshold,
                'nms_top_k': nms_top_k,
                'gaussian_sigma': gaussian_sigma,
                'use_gaussian': use_gaussian,
                'keep_top_k': keep_top_k,
                'normalized': normalized
            },
            outputs=outputs)
        output.stop_gradient = True

        if not return_index:
            index = None
        if not return_rois_num:
            rois_num = None
        return output, rois_num, index


# @torch.jit.not_to_static
def box_coder(prior_box,
              prior_box_var,
              target_box,
              code_type="encode_center_size",
              box_normalized=True,
              dim=0,
              name=None):
    r"""
    **Box Coder Layer**
    Encode/Decode the target bounding box with the priorbox information.

    The Encoding schema described below:
    .. math::
        ox = (tx - px) / pw / pxv
        oy = (ty - py) / ph / pyv
        ow = \log(\abs(tw / pw)) / pwv 
        oh = \log(\abs(th / ph)) / phv 
    The Decoding schema described below:

    .. math::

        ox = (pw * pxv * tx * + px) - tw / 2
        oy = (ph * pyv * ty * + py) - th / 2
        ow = \exp(pwv * tw) * pw + tw / 2
        oh = \exp(phv * th) * ph + th / 2   
    where `tx`, `ty`, `tw`, `th` denote the target box's center coordinates, 
    width and height respectively. Similarly, `px`, `py`, `pw`, `ph` denote 
    the priorbox's (anchor) center coordinates, width and height. `pxv`, 
    `pyv`, `pwv`, `phv` denote the variance of the priorbox and `ox`, `oy`, 
    `ow`, `oh` denote the encoded/decoded coordinates, width and height. 
    During Box Decoding, two modes for broadcast are supported. Say target 
    box has shape [N, M, 4], and the shape of prior box can be [N, 4] or 
    [M, 4]. Then prior box will broadcast to target box along the 
    assigned axis. 

    Args:
        prior_box(Tensor): Box list prior_box is a 2-D Tensor with shape 
            [M, 4] holds M boxes and data type is float32 or float64. Each box
            is represented as [xmin, ymin, xmax, ymax], [xmin, ymin] is the 
            left top coordinate of the anchor box, if the input is image feature
            map, they are close to the origin of the coordinate system. 
            [xmax, ymax] is the right bottom coordinate of the anchor box.       
        prior_box_var(List|Tensor|None): prior_box_var supports three types 
            of input. One is Tensor with shape [M, 4] which holds M group and 
            data type is float32 or float64. The second is list consist of 
            4 elements shared by all boxes and data type is float32 or float64. 
            Other is None and not involved in calculation. 
        target_box(Tensor): This input can be a 2-D LoDTensor with shape 
            [N, 4] when code_type is 'encode_center_size'. This input also can 
            be a 3-D Tensor with shape [N, M, 4] when code_type is 
            'decode_center_size'. Each box is represented as 
            [xmin, ymin, xmax, ymax]. The data type is float32 or float64. 
        code_type(str): The code type used with the target box. It can be
            `encode_center_size` or `decode_center_size`. `encode_center_size` 
            by default.
        box_normalized(bool): Whether treat the priorbox as a normalized box.
            Set true by default.
        axis(int): Which axis in PriorBox to broadcast for box decode, 
            for example, if axis is 0 and TargetBox has shape [N, M, 4] and 
            PriorBox has shape [M, 4], then PriorBox will broadcast to [N, M, 4]
            for decoding. It is only valid when code type is 
            `decode_center_size`. Set 0 by default. 
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Tensor:
        output_box(Tensor): When code_type is 'encode_center_size', the 
        output tensor of box_coder_op with shape [N, M, 4] representing the 
        result of N target boxes encoded with M Prior boxes and variances. 
        When code_type is 'decode_center_size', N represents the batch size 
        and M represents the number of decoded boxes.

    Examples:

        .. code-block:: python

            import torch
            from ppdettorch.modeling import ops
            torch.enable_static()
            # For encode
            prior_box_encode = torch.static.data(name='prior_box_encode',
                                  shape=[512, 4],
                                  dtype='float32')
            target_box_encode = torch.static.data(name='target_box_encode',
                                   shape=[81, 4],
                                   dtype='float32')
            output_encode = ops.box_coder(prior_box=prior_box_encode,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box_encode,
                                    code_type="encode_center_size")
            # For decode
            prior_box_decode = torch.static.data(name='prior_box_decode',
                                  shape=[512, 4],
                                  dtype='float32')
            target_box_decode = torch.static.data(name='target_box_decode',
                                   shape=[512, 81, 4],
                                   dtype='float32')
            output_decode = ops.box_coder(prior_box=prior_box_decode,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box_decode,
                                    code_type="decode_center_size",
                                    box_normalized=False,
                                    dim=1)
    """
    check_variable_and_dtype(prior_box, 'prior_box', ['float32', 'float64'],
                             'box_coder')
    check_variable_and_dtype(target_box, 'target_box', ['float32', 'float64'],
                             'box_coder')

    if in_dynamic_mode():
        if isinstance(prior_box_var, Variable):
            output_box = C_ops.box_coder(
                prior_box, prior_box_var, target_box, "code_type", code_type,
                "box_normalized", box_normalized, "axis", axis)

        elif isinstance(prior_box_var, list):
            output_box = C_ops.box_coder(
                prior_box, None, target_box, "code_type", code_type,
                "box_normalized", box_normalized, "axis", axis, "variance",
                prior_box_var)
        else:
            raise TypeError(
                "Input variance of box_coder must be Variable or list")
        return output_box
    else:
        helper = LayerHelper("box_coder", **locals())

        output_box = helper.create_variable_for_type_inference(
            dtype=prior_box.dtype)

        inputs = {"PriorBox": prior_box, "TargetBox": target_box}
        attrs = {
            "code_type": code_type,
            "box_normalized": box_normalized,
            "axis": axis
        }
        if isinstance(prior_box_var, Variable):
            inputs['PriorBoxVar'] = prior_box_var
        elif isinstance(prior_box_var, list):
            attrs['variance'] = prior_box_var
        else:
            raise TypeError(
                "Input variance of box_coder must be Variable or list")
        helper.append_op(
            type="box_coder",
            inputs=inputs,
            attrs=attrs,
            outputs={"OutputBox": output_box})
        return output_box


# @torch.jit.not_to_static
def generate_proposals(scores,
                       bbox_deltas,
                       im_shape,
                       anchors,
                       variances,
                       pre_nms_top_n=6000,
                       post_nms_top_n=1000,
                       nms_thresh=0.5,
                       min_size=0.1,
                       eta=1.0,
                       pixel_offset=False,
                       return_rois_num=False,
                       name=None):
    """
    **Generate proposal Faster-RCNN**
    This operation proposes RoIs according to each box with their
    probability to be a foreground object and 
    the box can be calculated by anchors. Bbox_deltais and scores
    to be an object are the output of RPN. Final proposals
    could be used to train detection net.
    For generating proposals, this operation performs following steps:
    1. Transposes and resizes scores and bbox_deltas in size of
       (H*W*A, 1) and (H*W*A, 4)
    2. Calculate box locations as proposals candidates. 
    3. Clip boxes to image
    4. Remove predicted boxes with small area. 
    5. Apply NMS to get final proposals as output.
    Args:
        scores(Tensor): A 4-D Tensor with shape [N, A, H, W] represents
            the probability for each box to be an object.
            N is batch size, A is number of anchors, H and W are height and
            width of the feature map. The data type must be float32.
        bbox_deltas(Tensor): A 4-D Tensor with shape [N, 4*A, H, W]
            represents the difference between predicted box location and
            anchor location. The data type must be float32.
        im_shape(Tensor): A 2-D Tensor with shape [N, 2] represents H, W, the
            origin image size or input size. The data type can be float32 or 
            float64.
        anchors(Tensor):   A 4-D Tensor represents the anchors with a layout
            of [H, W, A, 4]. H and W are height and width of the feature map,
            num_anchors is the box count of each position. Each anchor is
            in (xmin, ymin, xmax, ymax) format an unnormalized. The data type must be float32.
        variances(Tensor): A 4-D Tensor. The expanded variances of anchors with a layout of
            [H, W, num_priors, 4]. Each variance is in
            (xcenter, ycenter, w, h) format. The data type must be float32.
        pre_nms_top_n(float): Number of total bboxes to be kept per
            image before NMS. The data type must be float32. `6000` by default.
        post_nms_top_n(float): Number of total bboxes to be kept per
            image after NMS. The data type must be float32. `1000` by default.
        nms_thresh(float): Threshold in NMS. The data type must be float32. `0.5` by default.
        min_size(float): Remove predicted boxes with either height or
            width < min_size. The data type must be float32. `0.1` by default.
        eta(float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
            `adaptive_threshold = adaptive_threshold * eta` in each iteration.
        return_rois_num(bool): When setting True, it will return a 1D Tensor with shape [N, ] that includes Rois's 
            num of each image in one batch. The N is the image's num. For example, the tensor has values [4,5] that represents
            the first image has 4 Rois, the second image has 5 Rois. It only used in rcnn model. 
            'False' by default. 
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        tuple:
        A tuple with format ``(rpn_rois, rpn_roi_probs)``.
        - **rpn_rois**: The generated RoIs. 2-D Tensor with shape ``[N, 4]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
        - **rpn_roi_probs**: The scores of generated RoIs. 2-D Tensor with shape ``[N, 1]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.

    Examples:
        .. code-block:: python

            import torch
            from ppdettorch.modeling import ops
            torch.enable_static()
            scores = torch.static.data(name='scores', shape=[None, 4, 5, 5], dtype='float32')
            bbox_deltas = torch.static.data(name='bbox_deltas', shape=[None, 16, 5, 5], dtype='float32')
            im_shape = torch.static.data(name='im_shape', shape=[None, 2], dtype='float32')
            anchors = torch.static.data(name='anchors', shape=[None, 5, 4, 4], dtype='float32')
            variances = torch.static.data(name='variances', shape=[None, 5, 10, 4], dtype='float32')
            rois, roi_probs = ops.generate_proposals(scores, bbox_deltas,
                         im_shape, anchors, variances)
    """
    if in_dynamic_mode():
        assert return_rois_num, "return_rois_num should be True in dygraph mode."
        attrs = ('pre_nms_topN', pre_nms_top_n, 'post_nms_topN', post_nms_top_n,
                 'nms_thresh', nms_thresh, 'min_size', min_size, 'eta', eta,
                 'pixel_offset', pixel_offset)
        rpn_rois, rpn_roi_probs, rpn_rois_num = C_ops.generate_proposals_v2(
            scores, bbox_deltas, im_shape, anchors, variances, *attrs)
        if not return_rois_num:
            rpn_rois_num = None
        return rpn_rois, rpn_roi_probs, rpn_rois_num

    else:
        helper = LayerHelper('generate_proposals_v2', **locals())

        check_variable_and_dtype(scores, 'scores', ['float32'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(bbox_deltas, 'bbox_deltas', ['float32'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(im_shape, 'im_shape', ['float32', 'float64'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(anchors, 'anchors', ['float32'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(variances, 'variances', ['float32'],
                                 'generate_proposals_v2')

        rpn_rois = helper.create_variable_for_type_inference(
            dtype=bbox_deltas.dtype)
        rpn_roi_probs = helper.create_variable_for_type_inference(
            dtype=scores.dtype)
        outputs = {
            'RpnRois': rpn_rois,
            'RpnRoiProbs': rpn_roi_probs,
        }
        if return_rois_num:
            rpn_rois_num = helper.create_variable_for_type_inference(
                dtype='int32')
            rpn_rois_num.stop_gradient = True
            outputs['RpnRoisNum'] = rpn_rois_num

        helper.append_op(
            type="generate_proposals_v2",
            inputs={
                'Scores': scores,
                'BboxDeltas': bbox_deltas,
                'ImShape': im_shape,
                'Anchors': anchors,
                'Variances': variances
            },
            attrs={
                'pre_nms_topN': pre_nms_top_n,
                'post_nms_topN': post_nms_top_n,
                'nms_thresh': nms_thresh,
                'min_size': min_size,
                'eta': eta,
                'pixel_offset': pixel_offset
            },
            outputs=outputs)
        rpn_rois.stop_gradient = True
        rpn_roi_probs.stop_gradient = True
        if not return_rois_num:
            rpn_rois_num = None

        return rpn_rois, rpn_roi_probs, rpn_rois_num


def sigmoid_cross_entropy_with_logits(input,
                                      label,
                                      ignore_index=-100,
                                      normalize=False):
    output = F.binary_cross_entropy_with_logits(input, label, reduction='none')
    mask_tensor = torch.cast(label != ignore_index, 'float32')
    output = torch.multiply(output, mask_tensor)
    if normalize:
        sum_valid_mask = torch.sum(mask_tensor)
        output = output / sum_valid_mask
    return output


def smooth_l1(input, label, inside_weight=None, outside_weight=None,
              sigma=None):
    input_new = torch.multiply(input, inside_weight)
    label_new = torch.multiply(label, inside_weight)
    delta = 1 / (sigma * sigma)
    out = F.smooth_l1_loss(input_new, label_new, reduction='none', delta=delta)
    out = torch.multiply(out, outside_weight)
    out = out / delta
    out = torch.reshape(out, shape=[out.shape[0], -1])
    out = torch.sum(out, dim=1)
    return out


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    assert num_channels % groups == 0, 'num_channels should be divisible by groups'
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batch_size, -1, height, width)
    return x


def get_static_shape(tensor):
    shape = tensor.size()
    shape.stop_gradient = True
    return shape
