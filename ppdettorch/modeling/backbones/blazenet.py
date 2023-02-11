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
from ..shape_spec import ShapeSpec

__all__ = ['BlazeNet']


def hard_swish(x):
    return x * F.relu6(x + 3) / 6.


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 num_groups=1,
                 act='relu',
                 conv_lr=0.1,
                 conv_decay=0.,
                 norm_decay=0.,
                 norm_type='bn',
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias=False)

        if norm_type in ['bn', 'sync_bn']:
            self._batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        if self.act == "relu":
            x = F.relu(x)
        elif self.act == "relu6":
            x = F.relu6(x)
        elif self.act == 'leaky':
            x = F.leaky_relu(x)
        elif self.act == 'hard_swish':
            x = hard_swish(x)
        return x


class BlazeBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels1,
                 out_channels2,
                 double_channels=None,
                 stride=1,
                 use_5x5kernel=True,
                 act='relu',
                 name=None):
        super(BlazeBlock, self).__init__()
        assert stride in [1, 2]
        self.use_pool = not stride == 1
        self.use_double_block = double_channels is not None
        self.conv_dw = []
        if use_5x5kernel:
            conv_1 = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels1,
                kernel_size=5,
                stride=stride,
                padding=2,
                num_groups=out_channels1,
                name=name + "1_dw")
            self.add_module(name + "1_dw", conv_1)
            self.conv_dw.append(conv_1)
        else:
            conv_1 = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels1,
                kernel_size=3,
                stride=1,
                padding=1,
                num_groups=out_channels1,
                name=name + "1_dw_1")
            self.add_module(name + "1_dw_1", conv_1)
            self.conv_dw.append(conv_1)

            conv_2 = ConvBNLayer(
                in_channels=out_channels1,
                out_channels=out_channels1,
                kernel_size=3,
                stride=stride,
                padding=1,
                num_groups=out_channels1,
                name=name + "1_dw_2")
            self.add_module(name + "1_dw_2", conv_2)
            self.conv_dw.append(conv_2)
        self.act = act if self.use_double_block else None
        self.conv_pw = ConvBNLayer(
            in_channels=out_channels1,
            out_channels=out_channels2,
            kernel_size=1,
            stride=1,
            padding=0,
            act=self.act,
            name=name + "1_sep")
        if self.use_double_block:
            self.conv_dw2 = []
            if use_5x5kernel:
                conv_2 = ConvBNLayer(
                    in_channels=out_channels2,
                    out_channels=out_channels2,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    num_groups=out_channels2,
                    name=name + "2_dw")
                self.add_module(name + "2_dw", conv_2)
                self.conv_dw2.append(conv_2)
            else:
                conv_2 = ConvBNLayer(
                    in_channels=out_channels2,
                    out_channels=out_channels2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    num_groups=out_channels2,
                    name=name + "1_dw_1")
                self.add_module(name + "2_dw_1", conv_2)
                self.conv_dw2.append(conv_2)

                conv_3 = ConvBNLayer(
                    in_channels=out_channels2,
                    out_channels=out_channels2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    num_groups=out_channels2,
                    name=name + "2_dw_2")

                self.add_module(name + "2_dw_2", conv_3)
                self.conv_dw2.append(conv_3)
            self.conv_pw2 = ConvBNLayer(
                in_channels=out_channels2,
                out_channels=double_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                name=name + "2_sep")
        # shortcut
        if self.use_pool:
            shortcut_channel = double_channels or out_channels2
            self._shortcut = []
            temp_pool = nn.MaxPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
            self.add_module(name + '_shortcut_pool', temp_pool)
            self._shortcut.append(temp_pool)

            temp_pool = ConvBNLayer(
                in_channels=in_channels,
                out_channels=shortcut_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                name="shortcut" + name)
            self.add_module(name + '_shortcut_conv', temp_pool)
            self._shortcut.append(temp_pool)

    def forward(self, x):
        y = x
        for conv_dw_block in self.conv_dw:
            y = conv_dw_block(y)
        y = self.conv_pw(y)
        if self.use_double_block:
            for conv_dw2_block in self.conv_dw2:
                y = conv_dw2_block(y)
            y = self.conv_pw2(y)
        if self.use_pool:
            for shortcut in self._shortcut:
                x = shortcut(x)
        return F.relu(torch.add(x, y))


@register
@serializable
class BlazeNet(nn.Module):
    """
    BlazeFace, see https://arxiv.org/abs/1907.05047

    Args:
        blaze_filters (list): number of filter for each blaze block.
        double_blaze_filters (list): number of filter for each double_blaze block.
        use_5x5kernel (bool): whether or not filter size is 5x5 in depth-wise conv.
    """

    def __init__(
            self,
            blaze_filters=[[24, 24], [24, 24], [24, 48, 2], [48, 48], [48, 48]],
            double_blaze_filters=[[48, 24, 96, 2], [96, 24, 96], [96, 24, 96],
                                  [96, 24, 96, 2], [96, 24, 96], [96, 24, 96]],
            use_5x5kernel=True,
            act=None):
        super(BlazeNet, self).__init__()
        conv1_num_filters = blaze_filters[0][0]
        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=conv1_num_filters,
            kernel_size=3,
            stride=2,
            padding=1,
            name="conv1")
        in_channels = conv1_num_filters
        self.blaze_block = []
        self._out_channels = []
        for k, v in enumerate(blaze_filters):
            assert len(v) in [2, 3], \
                "blaze_filters {} not in [2, 3]"
            if len(v) == 2:
                temp_conv = BlazeBlock(in_channels,
                                       v[0],
                                       v[1],
                                       use_5x5kernel=use_5x5kernel,
                                       act=act,
                                       name='blaze_{}'.format(k))

                self.add_module('blaze_{}'.format(k), temp_conv)
                self.blaze_block.append(temp_conv)
            elif len(v) == 3:
                temp_conv = BlazeBlock(in_channels,
                                       v[0],
                                       v[1],
                                       stride=v[2],
                                       use_5x5kernel=use_5x5kernel,
                                       act=act,
                                       name='blaze_{}'.format(k))
                self.add_module('blaze_{}'.format(k), temp_conv)
                self.blaze_block.append(temp_conv)
            in_channels = v[1]

        for k, v in enumerate(double_blaze_filters):
            assert len(v) in [3, 4], \
                "blaze_filters {} not in [3, 4]"
            if len(v) == 3:
                temp_conv = BlazeBlock(in_channels, v[0],
                                       v[1],
                                       double_channels=v[2],
                                       use_5x5kernel=use_5x5kernel,
                                       act=act,
                                       name='double_blaze_{}'.format(k))
                self.add_module('double_blaze_{}'.format(k), temp_conv)
                self.blaze_block.append(temp_conv)
            elif len(v) == 4:
                temp_conv = BlazeBlock(in_channels,
                                       v[0],
                                       v[1],
                                       double_channels=v[2],
                                       stride=v[3],
                                       use_5x5kernel=use_5x5kernel,
                                       act=act,
                                       name='double_blaze_{}'.format(k))
                self.add_module('double_blaze_{}'.format(k), temp_conv)
                self.blaze_block.append(temp_conv)
            in_channels = v[2]
            self._out_channels.append(in_channels)

    def forward(self, inputs):
        outs = []
        y = self.conv1(inputs['image'])
        for block in self.blaze_block:
            y = block(y)
            outs.append(y)
        return [outs[-4], outs[-1]]

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=c)
            for c in [self._out_channels[-4], self._out_channels[-1]]
        ]
