#   Copyright (c) 2020 torchtorch Authors. All Rights Reserved.
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
import torch
import numpy as np


def bbox2delta(src_boxes, tgt_boxes, weights):
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * torch.log(tgt_w / src_w)
    dh = wh * torch.log(tgt_h / src_h)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    return deltas


def delta2bbox(deltas, boxes, weights):
    clip_scale = math.log(1000.0 / 16)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    # Prevent sending too large values into torch.exp()
    dw = torch.clip(dw, max=clip_scale)
    dh = torch.clip(dh, max=clip_scale)

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = torch.stack(pred_boxes, dim=-1)

    return pred_boxes


def expand_bbox(bboxes, scale):
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    bboxes_exp = np.zeros(bboxes.shape, dtype=np.float32)
    bboxes_exp[:, 0] = x_c - w_half
    bboxes_exp[:, 2] = x_c + w_half
    bboxes_exp[:, 1] = y_c - h_half
    bboxes_exp[:, 3] = y_c + h_half

    return bboxes_exp


def clip_bbox(boxes, im_shape):
    h, w = im_shape[0], im_shape[1]
    x1 = boxes[:, 0].clip(0, w)
    y1 = boxes[:, 1].clip(0, h)
    x2 = boxes[:, 2].clip(0, w)
    y2 = boxes[:, 3].clip(0, h)
    return torch.stack([x1, y1, x2, y2], dim=1)


def nonempty_bbox(boxes, min_size=0, return_mask=False):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = torch.logical_and(h > min_size, w > min_size)
    if return_mask:
        return mask
    keep = torch.nonzero(mask).flatten()
    return keep


def bbox_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def bbox_overlaps(boxes1, boxes2):
    """
    Calculate overlaps between boxes1 and boxes2

    Args:
        boxes1 (Tensor): boxes with shape [M, 4]
        boxes2 (Tensor): boxes with shape [N, 4]

    Return:
        overlaps (Tensor): overlaps between boxes1 and boxes2 with shape [M, N]
    """
    M = boxes1.shape[0]
    N = boxes2.shape[0]
    if M * N == 0:
        return torch.zeros([M, N])
    area1 = bbox_area(boxes1)
    area2 = bbox_area(boxes2)

    xy_max = torch.minimum(
        torch.unsqueeze(boxes1, 1)[:, :, 2:], boxes2[:, 2:])
    xy_min = torch.maximum(
        torch.unsqueeze(boxes1, 1)[:, :, :2], boxes2[:, :2])
    width_height = xy_max - xy_min
    width_height = width_height.clip(min=0)
    inter = width_height.prod(dim=2)

    overlaps = torch.where(inter > 0, inter /
                           (torch.unsqueeze(area1, 1) + area2 - inter),
                           torch.zeros_like(inter))
    return overlaps


def batch_bbox_overlaps(bboxes1,
                        bboxes2,
                        mode='iou',
                        is_aligned=False,
                        eps=1e-6):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert mode in ['iou', 'iof', 'giou'], 'Unsupported mode {}'.format(mode)
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2] if bboxes1.shape[0] > 0 else 0
    cols = bboxes2.shape[-2] if bboxes2.shape[0] > 0 else 0
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return torch.full(batch_shape + (rows,), 1)
        else:
            return torch.full(batch_shape + (rows, cols), 1)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    if is_aligned:
        lt = torch.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [B, rows, 2]
        rb = torch.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [B, rows, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, 2]
        overlap = wh[:, 0] * wh[:, 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.minimum(bboxes1[:, :2], bboxes2[:, :2])
            enclosed_rb = torch.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    else:
        lt = torch.maximum(bboxes1[:, :2].reshape([rows, 1, 2]),
                           bboxes2[:, :2])  # [B, rows, cols, 2]
        rb = torch.minimum(bboxes1[:, 2:].reshape([rows, 1, 2]),
                           bboxes2[:, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]

        if mode in ['iou', 'giou']:
            union = area1.reshape([rows, 1]) \
                    + area2.reshape([1, cols]) - overlap
        else:
            union = area1[:, None]
        if mode == 'giou':
            enclosed_lt = torch.minimum(bboxes1[:, :2].reshape([rows, 1, 2]),
                                        bboxes2[:, :2])
            enclosed_rb = torch.maximum(bboxes1[:, 2:].reshape([rows, 1, 2]),
                                        bboxes2[:, 2:])

    eps = torch.FloatTensor([eps])
    union = torch.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clip(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]
    enclose_area = torch.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return 1 - gious


def xywh2xyxy(box):
    x, y, w, h = box
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return [x1, y1, x2, y2]


def xywh2xyxy_v2(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape
    print(f"current_dim: {current_dim} -original_shape: {original_shape}")

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def make_grid(h, w, dtype):
    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    return torch.stack((xv, yv), 2).cast(dtype=dtype)


def decode_yolo(box, anchor, downsample_ratio):
    """decode yolo box

    Args:
        box (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        anchor (list): anchor with the shape [na, 2]
        downsample_ratio (int): downsample ratio, default 32
        scale (float): scale, default 1.

    Return:
        box (list): decoded box, [x, y, w, h], all have the shape [b, na, h, w, 1]
    """
    x, y, w, h = box
    na, grid_h, grid_w = x.shape[1:4]
    grid = make_grid(grid_h, grid_w, x.dtype).reshape((1, 1, grid_h, grid_w, 2))
    x1 = (x + grid[:, :, :, :, 0:1]) / grid_w
    y1 = (y + grid[:, :, :, :, 1:2]) / grid_h

    anchor = torch.FloatTensor(anchor)
    # anchor = torch.cast(anchor, x.dtype)
    anchor = anchor.reshape((1, na, 1, 1, 2))
    w1 = torch.exp(w) * anchor[:, :, :, :, 0:1] / (downsample_ratio * grid_w)
    h1 = torch.exp(h) * anchor[:, :, :, :, 1:2] / (downsample_ratio * grid_h)

    return [x1, y1, w1, h1]


def batch_iou_similarity(box1, box2, eps=1e-9):
    """Calculate iou of box1 and box2 in batch

    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def bbox_iou(box1,
             box2,
             x1y1x2y2=True,
             giou=False,
             diou=False,
             ciou=False,
             eps=1e-9):
    """calculate the iou of box1 and box2

    Args:
        box1 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        box2 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero

    Return:
        iou (Tensor): iou of box1 and box1, with the shape [b, na, h, w, 1]
    """
    if x1y1x2y2:
        px1, py1, px2, py2 = box1
        gx1, gy1, gx2, gy2 = box2
    else:  # transform from xywh to xyxy
        px1, px2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        py1, py2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        gx1, gx2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        gy1, gy2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    x1 = torch.maximum(px1, gx1)
    y1 = torch.maximum(py1, gy1)
    x2 = torch.minimum(px2, gx2)
    y2 = torch.minimum(py2, gy2)

    overlap = ((x2 - x1).clip(0)) * ((y2 - y1).clip(0))

    area1 = (px2 - px1) * (py2 - py1)
    area1 = area1.clip(0)

    area2 = (gx2 - gx1) * (gy2 - gy1)
    area2 = area2.clip(0)

    union = area1 + area2 - overlap + eps
    iou = overlap / union

    if giou or ciou or diou:
        # convex w, h
        cw = torch.maximum(px2, gx2) - torch.minimum(px1, gx1)
        ch = torch.maximum(py2, gy2) - torch.minimum(py1, gy1)
        if giou:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        else:
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + eps
            # center distance
            rho2 = ((px1 + px2 - gx1 - gx2) ** 2 + (py1 + py2 - gy1 - gy2) ** 2) / 4
            if diou:
                return iou - rho2 / c2
            else:
                w1, h1 = px2 - px1, py2 - py1 + eps
                w2, h2 = gx2 - gx1, gy2 - gy1 + eps
                delta = torch.atan(w1 / h1) - torch.atan(w2 / h2)
                v = (4 / math.pi ** 2) * torch.pow(delta, 2)
                alpha = v / (1 + eps - iou + v)
                alpha.stop_gradient = True
                return iou - (rho2 / c2 + v * alpha)
    else:
        return iou


def bbox_iou_np_expand(box1, box2, x1y1x2y2=True, eps=1e-16):
    """
    Calculate the iou of box1 and box2 with numpy.

    Args:
        box1 (ndarray): [N, 4]
        box2 (ndarray): [M, 4], usually N != M
        x1y1x2y2 (bool): whether in x1y1x2y2 stype, default True
        eps (float): epsilon to avoid divide by zero
    Return:
        iou (ndarray): iou of box1 and box2, [N, M]
    """
    N, M = len(box1), len(box2)  # usually N != M
    if x1y1x2y2:
        b1_x1, b1_y1 = box1[:, 0], box1[:, 1]
        b1_x2, b1_y2 = box1[:, 2], box1[:, 3]
        b2_x1, b2_y1 = box2[:, 0], box2[:, 1]
        b2_x2, b2_y2 = box2[:, 2], box2[:, 3]
    else:
        # cxcywh style
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = np.zeros((N, M), dtype=np.float32)
    inter_rect_y1 = np.zeros((N, M), dtype=np.float32)
    inter_rect_x2 = np.zeros((N, M), dtype=np.float32)
    inter_rect_y2 = np.zeros((N, M), dtype=np.float32)
    for i in range(len(box2)):
        inter_rect_x1[:, i] = np.maximum(b1_x1, b2_x1[i])
        inter_rect_y1[:, i] = np.maximum(b1_y1, b2_y1[i])
        inter_rect_x2[:, i] = np.minimum(b1_x2, b2_x2[i])
        inter_rect_y2[:, i] = np.minimum(b1_y2, b2_y2[i])
    # Intersection area
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(
        inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = np.repeat(
        ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).reshape(-1, 1), M, dim=-1)
    b2_area = np.repeat(
        ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).reshape(1, -1), N, dim=0)

    ious = inter_area / (b1_area + b2_area - inter_area + eps)
    return ious


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=
    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clip(min=0, max=max_dis - eps)
        top = top.clip(min=0, max=max_dis - eps)
        right = right.clip(min=0, max=max_dis - eps)
        bottom = bottom.clip(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox_center(boxes):
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)


def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = torch.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.concat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox,
                               torch.zeros_like(out_bbox))
    return out_bbox


def delta2bbox_v2(rois,
                  deltas,
                  means=(0.0, 0.0, 0.0, 0.0),
                  stds=(1.0, 1.0, 1.0, 1.0),
                  max_shape=None,
                  wh_ratio_clip=16.0 / 1000.0,
                  ctr_clip=None):
    """Transform network output(delta) to bboxes.
    Based on https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/
             bbox/coder/delta_xywh_bbox_coder.py
    Args:
        rois (Tensor): shape [..., 4], base bboxes, typical examples include
            anchor and rois
        deltas (Tensor): shape [..., 4], offset relative to base bboxes
        means (list[float]): the mean that was used to normalize deltas,
            must be of size 4
        stds (list[float]): the std that was used to normalize deltas,
            must be of size 4
        max_shape (list[float] or None): height and width of image, will be
            used to clip bboxes if not None
        wh_ratio_clip (float): to clip delta wh of decoded bboxes
        ctr_clip (float or None): whether to clip delta xy of decoded bboxes
    """
    if rois.size == 0:
        return torch.empty_like(rois)
    means = torch.FloatTensor(means)
    stds = torch.FloatTensor(stds)
    deltas = deltas * stds + means

    dxy = deltas[..., :2]
    dwh = deltas[..., 2:]

    pxy = (rois[..., :2] + rois[..., 2:]) * 0.5
    pwh = rois[..., 2:] - rois[..., :2]
    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if ctr_clip is not None:
        dxy_wh = torch.clip(dxy_wh, max=ctr_clip, min=-ctr_clip)
        dwh = torch.clip(dwh, max=max_ratio)
    else:
        dwh = dwh.clip(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.concat([x1y1, x2y2], dim=-1)
    if max_shape is not None:
        bboxes[..., 0::2] = bboxes[..., 0::2].clip(min=0, max=max_shape[1])
        bboxes[..., 1::2] = bboxes[..., 1::2].clip(min=0, max=max_shape[0])
    return bboxes


def bbox2delta_v2(src_boxes,
                  tgt_boxes,
                  means=(0.0, 0.0, 0.0, 0.0),
                  stds=(1.0, 1.0, 1.0, 1.0)):
    """Encode bboxes to deltas.
    Modified from ppdettorch.modeling.bbox_utils.bbox2delta.
    Args:
        src_boxes (Tensor[..., 4]): base bboxes
        tgt_boxes (Tensor[..., 4]): target bboxes
        means (list[float]): the mean that will be used to normalize delta
        stds (list[float]): the std that will be used to normalize delta
    """
    if src_boxes.size == 0:
        return torch.empty_like(src_boxes)
    src_w = src_boxes[..., 2] - src_boxes[..., 0]
    src_h = src_boxes[..., 3] - src_boxes[..., 1]
    src_ctr_x = src_boxes[..., 0] + 0.5 * src_w
    src_ctr_y = src_boxes[..., 1] + 0.5 * src_h

    tgt_w = tgt_boxes[..., 2] - tgt_boxes[..., 0]
    tgt_h = tgt_boxes[..., 3] - tgt_boxes[..., 1]
    tgt_ctr_x = tgt_boxes[..., 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[..., 1] + 0.5 * tgt_h

    dx = (tgt_ctr_x - src_ctr_x) / src_w
    dy = (tgt_ctr_y - src_ctr_y) / src_h
    dw = torch.log(tgt_w / src_w)
    dh = torch.log(tgt_h / src_h)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)  # [n, 4]
    means = torch.FloatTensor(means, place=src_boxes.place)
    stds = torch.FloatTensor(stds, place=src_boxes.place)
    deltas = (deltas - means) / stds
    return deltas


def iou_similarity(box1, box2, eps=1e-10):
    """Calculate iou of box1 and box2

    Args:
        box1 (Tensor): box with the shape [M1, 4]
        box2 (Tensor): box with the shape [M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [M1, M2]
    """
    box1 = box1.unsqueeze(1)  # [M1, 4] -> [M1, 1, 4]
    box2 = box2.unsqueeze(0)  # [M2, 4] -> [1, M2, 4]
    px1y1, px2y2 = box1[:, :, 0:2], box1[:, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, 0:2], box2[:, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def yolo_box_pytorch(x,
                     img_size,
                     anchors,
                     class_num,
                     conf_thresh,
                     downsample_ratio,
                     clip_bbox=True,
                     name=None,
                     scale_x_y=1.,
                     iou_aware=False,
                     iou_aware_factor=0.5):
    r"""

    This operator generates YOLO detection boxes from output of YOLOv3 network.

    The output of previous network is in shape [N, C, H, W], while H and W
    should be the same, H and W specify the grid size, each grid point predict
    given number boxes, this given number, which following will be represented as S,
    is specified by the number of anchors. In the second dimension(the channel
    dimension), C should be equal to S * (5 + class_num) if :attr:`iou_aware` is false,
    otherwise C should be equal to S * (6 + class_num). class_num is the object
    category number of source dataset(such as 80 in coco dataset), so the
    second(channel) dimension, apart from 4 box location coordinates x, y, w, h,
    also includes confidence score of the box and class one-hot key of each anchor
    box.

    Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box
    predictions should be as follows:

    $$
    b_x = \\sigma(t_x) + c_x
    $$
    $$
    b_y = \\sigma(t_y) + c_y
    $$
    $$
    b_w = p_w e^{t_w}
    $$
    $$
    b_h = p_h e^{t_h}
    $$

    in the equation above, :math:`c_x, c_y` is the left top corner of current grid
    and :math:`p_w, p_h` is specified by anchors.

    The logistic regression value of the 5th channel of each anchor prediction boxes
    represents the confidence score of each prediction box, and the logistic
    regression value of the last :attr:`class_num` channels of each anchor prediction
    boxes represents the classifcation scores. Boxes with confidence scores less than
    :attr:`conf_thresh` should be ignored, and box final scores is the product of
    confidence scores and classification scores.

    $$
    score_{pred} = score_{conf} * score_{class}
    $$

    where the confidence scores follow the formula bellow

    .. math::

        score_{conf} = \begin{case}
                         obj, \text{if } iou_aware == flase \\
                         obj^{1 - iou_aware_factor} * iou^{iou_aware_factor}, \text{otherwise}
                       \end{case}

    Args:
        x (Tensor): The input tensor of YoloBox operator is a 4-D tensor with
                      shape of [N, C, H, W]. The second dimension(C) stores box
                      locations, confidence score and classification one-hot keys
                      of each anchor box. Generally, X should be the output of
                      YOLOv3 network. The data type is float32 or float64.
        img_size (Tensor): The image size tensor of YoloBox operator, This is a
                           2-D tensor with shape of [N, 2]. This tensor holds
                           height and width of each input image used for resizing
                           output box in input image scale. The data type is int32.
        anchors (list|tuple): The anchor width and height, it will be parsed pair
                              by pair.
        class_num (int): The number of classes.
        conf_thresh (float): The confidence scores threshold of detection boxes.
                             Boxes with confidence scores under threshold should
                             be ignored.
        downsample_ratio (int): The downsample ratio from network input to
                                :attr:`yolo_box` operator input, so 32, 16, 8
                                should be set for the first, second, and thrid
                                :attr:`yolo_box` layer.
        clip_bbox (bool): Whether clip output bonding box in :attr:`img_size`
                          boundary. Default true.
        scale_x_y (float): Scale the center point of decoded bounding box.
                           Default 1.0
        name (string): The default value is None.  Normally there is no need
                       for user to set this property.  For more information,
                       please refer to :ref:`api_guide_Name`
        iou_aware (bool): Whether use iou aware. Default false
        iou_aware_factor (float): iou aware factor. Default 0.5

    Returns:
        Tensor: A 3-D tensor with shape [N, M, 4], the coordinates of boxes,
        and a 3-D tensor with shape [N, M, :attr:`class_num`], the classification
        scores of boxes.

    Raises:
        TypeError: Input x of yolov_box must be Tensor
        TypeError: Attr anchors of yolo box must be list or tuple
        TypeError: Attr class_num of yolo box must be an integer
        TypeError: Attr conf_thresh of yolo box must be a float number

    Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        x = np.random.random([2, 14, 8, 8]).astype('float32')
        img_size = np.ones((2, 2)).astype('int32')

        x = paddle.to_tensor(x)
        img_size = paddle.to_tensor(img_size)

        boxes, scores = paddle.vision.ops.yolo_box(x,
                                                   img_size=img_size,
                                                   anchors=[10, 13, 16, 30],
                                                   class_num=2,
                                                   conf_thresh=0.01,
                                                   downsample_ratio=8,
                                                   clip_bbox=True,
                                                   scale_x_y=1.)
    """
    run_device = x.device
    (n, c, h, w) = x.shape
    an_num = int((len(anchors) // 2))

    bias_x_y = (-0.5) * (scale_x_y - 1.0)
    input_h = downsample_ratio * h
    input_w = downsample_ratio * w
    if iou_aware:
        ioup = x[:, :an_num, :, :]
        ioup = ioup.unsqueeze(-1)
        x = x[:, an_num:, :, :]

    x = x.view((n, an_num, (5 + class_num), h, w)).permute(0, 1, 3, 4, 2).contiguous()

    pred_box = torch.clone(x[..., :4])

    # grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    # grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))

    grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij')

    grid_y = grid_y.to(run_device)
    grid_x = grid_x.to(run_device)

    pred_box[..., 0] = (
                               (grid_x + (pred_box[..., 0].sigmoid() * scale_x_y)) + bias_x_y
                       ) / w
    pred_box[..., 1] = (
                               (grid_y + (pred_box[..., 1].sigmoid() * scale_x_y)) + bias_x_y
                       ) / h
    anchors = [
        (anchors[i], anchors[(i + 1)]) for i in range(0, len(anchors), 2)
    ]
    anchors_s = np.array(
        [((an_w / input_w), (an_h / input_h)) for (an_w, an_h) in anchors]
    )
    anchors_s = torch.tensor(anchors_s, device=run_device)
    anchor_w = anchors_s[:, 0:1].reshape(1, an_num, 1, 1)
    anchor_h = anchors_s[:, 1:2].reshape(1, an_num, 1, 1)

    pred_box[..., 2] = torch.exp(pred_box[..., 2]) * anchor_w
    pred_box[..., 3] = torch.exp(pred_box[..., 3]) * anchor_h
    if iou_aware:
        pred_conf = (x[..., 4:5].sigmoid() ** (1 - iou_aware_factor)) * (
                torch.sigmoid(ioup) ** iou_aware_factor
        )
    else:
        pred_conf = x[..., 4:5].sigmoid()

    pred_conf[(pred_conf < conf_thresh)] = 0.0
    pred_score = x[..., 5:].sigmoid() * pred_conf
    pred_box = pred_box * (pred_conf > 0.0)
    pred_box = pred_box.reshape(n, -1, 4)
    (pred_box[..., :2], pred_box[..., 2:4]) = (
        (pred_box[..., :2] - (pred_box[..., 2:4] / 2.0)),
        (pred_box[..., :2] + (pred_box[..., 2:4] / 2.0)),
    )

    # pred_box = xywh2xyxy_v2(pred_box[:, :4])

    pred_box[..., 0] = pred_box[..., 0] * img_size[:, 1]
    pred_box[..., 1] = pred_box[..., 1] * img_size[:, 0]
    pred_box[..., 2] = pred_box[..., 2] * img_size[:, 1]
    pred_box[..., 3] = pred_box[..., 3] * img_size[:, 0]
    if clip_bbox:
        for i in range(len(pred_box)):
            pred_box[i, :, 0] = torch.clamp(pred_box[i, :, 0], 0, torch.inf)
            pred_box[i, :, 1] = torch.clamp(pred_box[i, :, 1], 0, torch.inf)
            pred_box[i, :, 2] = torch.clamp(
                pred_box[i, :, 2], (-torch.inf), (img_size[(i, 1)] - 1)
            )
            pred_box[i, :, 3] = torch.clamp(
                pred_box[i, :, 3], (-torch.inf), (img_size[(i, 0)] - 1)
            )
    return (pred_box, pred_score.view(n, -1, class_num))


def ssd_prior_box_np(
        input,
        image,
        min_sizes,
        aspect_ratios=[1.0],
        variances=[0.1, 0.1, 0.2, 0.2],
        max_sizes=None,
        flip=False,
        clip=False,
        step_w=0,
        step_h=0,
        offset=0.5,
        min_max_aspect_ratios_order=False,
        name=None,
):
    r"""

    This op generates prior boxes for SSD(Single Shot MultiBox Detector) algorithm.

    Each position of the input produce N prior boxes, N is determined by
    the count of min_sizes, max_sizes and aspect_ratios, The size of the
    box is in range(min_size, max_size) interval, which is generated in
    sequence according to the aspect_ratios.

    Args:
       input (Tensor): 4-D tensor(NCHW), the data type should be float32 or float64.
       image (Tensor): 4-D tensor(NCHW), the input image data of PriorBoxOp,
            the data type should be float32 or float64.
       min_sizes (list|tuple|float): the min sizes of generated prior boxes.
       max_sizes (list|tuple|None, optional): the max sizes of generated prior boxes.
            Default: None, means [] and will not be used.
       aspect_ratios (list|tuple|float, optional): the aspect ratios of generated
            prior boxes. Default: [1.0].
       variance (list|tuple, optional): the variances to be encoded in prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       flip (bool): Whether to flip aspect ratios. Default:False.
       clip (bool): Whether to clip out-of-boundary boxes. Default: False.
       steps (list|tuple, optional): Prior boxes steps across width and height, If
            steps[0] equals to 0.0 or steps[1] equals to 0.0, the prior boxes steps across
            height or weight of the input will be automatically calculated.
            Default: [0., 0.]
       offset (float, optional)): Prior boxes center offset. Default: 0.5
       min_max_aspect_ratios_order (bool, optional): If set True, the output prior box is
            in order of [min, max, aspect_ratios], which is consistent with
            Caffe. Please note, this order affects the weights order of
            convolution layer followed by and does not affect the final
            detection results. Default: False.
       name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: the output prior boxes and the expanded variances of PriorBox.
            The prior boxes is a 4-D tensor, the layout is [H, W, num_priors, 4],
            num_priors is the total box count of each position of input.
            The expanded variances is a 4-D tensor, same shape as the prior boxes.

    Examples:
        .. code-block:: python

            import paddle

            input = paddle.rand((1, 3, 6, 9), dtype=paddle.float32)
            image = paddle.rand((1, 3, 9, 12), dtype=paddle.float32)

            box, var = paddle.vision.ops.prior_box(
                input=input,
                image=image,
                min_sizes=[2.0, 4.0],
                clip=True,
                flip=True)

    """

    _, _, layer_h, layer_w = input.shape
    _, _, image_h, image_w = image.shape

    epsilon = 1e-6
    real_aspect_ratios = [1]
    for ar in aspect_ratios:
        already_exist = False
        for exist_ar in real_aspect_ratios:
            if abs(ar - exist_ar) < epsilon:
                already_exist = True
                break
        if not already_exist:
            real_aspect_ratios.append(ar)
            if flip:
                real_aspect_ratios.append(1.0 / ar)

    if step_w == 0 or step_h == 0:
        step_w = image_w / layer_w
        step_h = image_h / layer_h

    num_priors = len(real_aspect_ratios) * len(min_sizes)
    if max_sizes is None:
        max_sizes = []

    if len(max_sizes) > 0:
        num_priors += len(max_sizes)

    out_dim = (layer_h, layer_w, num_priors, 4)
    out_boxes = np.zeros(out_dim).astype('float32')

    for h in range(layer_h):
        for w in range(layer_w):
            c_x = (w + offset) * step_w
            c_y = (h + offset) * step_h
            idx = 0
            for s in range(len(min_sizes)):
                min_size = min_sizes[s]
                if not min_max_aspect_ratios_order:
                    # rest of priors
                    for r in range(len(real_aspect_ratios)):
                        ar = real_aspect_ratios[r]
                        c_w = min_size * math.sqrt(ar) / 2
                        c_h = (min_size / math.sqrt(ar)) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / image_w,
                            (c_y - c_h) / image_h,
                            (c_x + c_w) / image_w,
                            (c_y + c_h) / image_h,
                        ]
                        idx += 1

                    if len(max_sizes) > 0:
                        max_size = max_sizes[s]
                        # second prior: aspect_ratio = 1,
                        c_w = c_h = math.sqrt(min_size * max_size) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / image_w,
                            (c_y - c_h) / image_h,
                            (c_x + c_w) / image_w,
                            (c_y + c_h) / image_h,
                        ]
                        idx += 1
                else:
                    c_w = c_h = min_size / 2.0
                    out_boxes[h, w, idx, :] = [
                        (c_x - c_w) / image_w,
                        (c_y - c_h) / image_h,
                        (c_x + c_w) / image_w,
                        (c_y + c_h) / image_h,
                    ]
                    idx += 1
                    if len(max_sizes) > 0:
                        max_size = max_sizes[s]
                        # second prior: aspect_ratio = 1,
                        c_w = c_h = math.sqrt(min_size * max_size) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / image_w,
                            (c_y - c_h) / image_h,
                            (c_x + c_w) / image_w,
                            (c_y + c_h) / image_h,
                        ]
                        idx += 1

                    # rest of priors
                    for r in range(len(real_aspect_ratios)):
                        ar = real_aspect_ratios[r]
                        if abs(ar - 1.0) < 1e-6:
                            continue
                        c_w = min_size * math.sqrt(ar) / 2
                        c_h = (min_size / math.sqrt(ar)) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / image_w,
                            (c_y - c_h) / image_h,
                            (c_x + c_w) / image_w,
                            (c_y + c_h) / image_h,
                        ]
                        idx += 1

    # clip the prior's coordidate such that it is within[0, 1]
    if clip:
        out_boxes = np.clip(out_boxes, 0.0, 1.0)
    # set the variance.
    out_var = np.tile(
        variances, (layer_h, layer_w, num_priors, 1)
    )

    return out_boxes, out_var
