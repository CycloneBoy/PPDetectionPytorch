# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, MaxPool2d

import copy

from ppdettorch.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['PPHGNetV2']


class LearnableAffineBlock(nn.Module):
    def __init__(self,
                 scale_value=1.0,
                 bias_value=0.0,
                 lr_mult=1.0,
                 lab_lr=0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros([1, ]))
        self.register_parameter("scale", self.scale)
        self.bias = nn.Parameter(torch.zeros([1, ]))
        self.register_parameter("bias", self.bias)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 use_act=True,
                 use_lab=False,
                 lr_mult=1.0):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding
            if isinstance(padding, str) else (kernel_size - 1) // 2,
            groups=groups,
            bias=False)
        self.bn = BatchNorm2d(out_channels)
        if self.use_act:
            self.act = ReLU()
            if self.use_lab:
                self.lab = LearnableAffineBlock(lr_mult=lr_mult)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_lab=False,
                 lr_mult=1.0):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
            use_lab=use_lab,
            lr_mult=lr_mult)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 use_lab=False,
                 lr_mult=1.0):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding="same",
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding="same",
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=1, ceil_mode=True, padding=1)

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)[:, :, :x.shape[2], :x.shape[3]]
        x = torch.concat([x1, x2], 1)
        x = self.stem3(x)
        x = self.stem4(x)

        return x


class HG_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 layer_num=6,
                 identity=False,
                 light_block=True,
                 use_lab=False,
                 lr_mult=1.0):
        super().__init__()
        self.identity = identity

        self.layers = nn.ModuleList()
        block_type = "LightConvBNAct" if light_block else "ConvBNAct"
        for i in range(layer_num):
            self.layers.append(
                eval(block_type)(in_channels=in_channels
                if i == 0 else mid_channels,
                                 out_channels=mid_channels,
                                 stride=1,
                                 kernel_size=kernel_size,
                                 use_lab=use_lab,
                                 lr_mult=lr_mult))
        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.aggregation_excitation_conv = ConvBNAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.concat(output, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x += identity
        return x


class HG_Stage(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 block_num,
                 layer_num=6,
                 downsample=True,
                 light_block=True,
                 kernel_size=3,
                 use_lab=False,
                 lr_mult=1.0):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab,
                lr_mult=lr_mult)

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=False if i == 0 else True,
                    light_block=light_block,
                    use_lab=use_lab,
                    lr_mult=lr_mult))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


def _freeze_norm(m: nn.BatchNorm2d):
    global_stats = True
    norm = nn.BatchNorm2d(m.num_features, track_running_stats=global_stats)
    for param in norm.parameters():
        param.requires_grad = False
    return norm


def reset_bn(model: nn.Module, reset_func=_freeze_norm):
    if isinstance(model, nn.BatchNorm2d):
        model = reset_func(model)
    else:
        for name, child in model.named_children():
            _child = reset_bn(child, reset_func)
            if _child is not child:
                setattr(model, name, _child)
    return model


@register
@serializable
class PPHGNetV2(nn.Module):
    """
    PPHGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of PPHGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Module. Specific PPHGNetV2 model depends on args.
    """

    arch_configs = {
        'L': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            }
        },
        'X': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            }
        }
    }

    def __init__(self,
                 arch,
                 use_lab=False,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[arch]['stem_channels']
        stage_config = self.arch_configs[arch]['stage_config']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
            use_lab=use_lab,
            lr_mult=lr_mult_list[0])

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = \
                stage_config[
                    k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    lr_mult=lr_mult_list[i + 1]))

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            reset_bn(self, reset_func=_freeze_norm)

        self._init_weights()

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.stop_gradient = True

    def _init_weights(self):
        pass

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
