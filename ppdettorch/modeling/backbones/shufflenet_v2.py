# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppdettorch.core.workspace import register, serializable
from numbers import Integral
from ..shape_spec import ShapeSpec
from ppdettorch.modeling.ops import channel_shuffle

__all__ = ['ShuffleNetV2']


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self._batch_norm = nn.BatchNorm2d(out_channels, track_running_stats=True)
        if act == "hard_swish":
            act = 'hardswish'
        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act:
            y = getattr(F, self.act)(y)
        return y


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None)
        self._conv_linear = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1, x2 = torch.split(
            inputs,
            split_size_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            dim=1)
        x2 = self._conv_pw(x2)
        x2 = self._conv_dw(x2)
        x2 = self._conv_linear(x2)
        out = torch.concat([x1, x2], dim=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act="relu"):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None)
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        # branch2
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None)
        self._conv_linear_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.concat([x1, x2], dim=1)

        return channel_shuffle(out, 2)


@register
@serializable
class ShuffleNetV2(nn.Module):
    def __init__(self, scale=1.0, act="relu", feature_maps=[5, 13, 17]):
        super(ShuffleNetV2, self).__init__()
        self.scale = scale
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        self.feature_maps = feature_maps
        stage_repeats = [4, 8, 4]

        if scale == 0.25:
            stage_out_channels = [-1, 24, 24, 48, 96, 512]
        elif scale == 0.33:
            stage_out_channels = [-1, 24, 32, 64, 128, 512]
        elif scale == 0.5:
            stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif scale == 1.0:
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif scale == 1.5:
            stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif scale == 2.0:
            stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError("This scale size:[" + str(scale) +
                                      "] is not implemented!")
        self._out_channels = []
        self._feature_idx = 0
        # 1. conv1
        self._conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            act=act)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        # 2. bottleneck sequences
        self._block_list = []
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                if i == 0:
                    block = InvertedResidualDS(
                        in_channels=stage_out_channels[stage_id + 1],
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=2,
                        act=act)
                    self.add_module(name=str(stage_id + 2) + '_' + str(i + 1), module=block)
                else:
                    block = InvertedResidual(
                        in_channels=stage_out_channels[stage_id + 2],
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=1,
                        act=act)
                    self.add_module(name=str(stage_id + 2) + '_' + str(i + 1), module=block)
                self._block_list.append(block)
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2],
                                          self._feature_idx, self.feature_maps)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self, inputs):
        y = self._conv1(inputs['image'])
        y = self._max_pool(y)
        outs = []
        for i, inv in enumerate(self._block_list):
            y = inv(y)
            if i + 2 in self.feature_maps:
                outs.append(y)

        return outs

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
