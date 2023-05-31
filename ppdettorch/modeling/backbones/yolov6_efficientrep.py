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
"""
This code is based on https://github.com/meituan/YOLOv6
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ppdettorch.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['EfficientRep', 'CSPBepBackbone', 'Lite_EffiBackbone']


def get_activation(name="silu"):
    if name == "silu":
        module = nn.SiLU()
    elif name == "relu":
        module = nn.ReLU()
    elif name in ["LeakyReLU", 'leakyrelu', 'lrelu']:
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


class BaseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)  # silu

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.training:
            y = self.act(x)
        else:
            if isinstance(self.act, nn.SiLU):
                self.act = SiLU()
            y = self.act(x)
        return y


class RepConv(nn.Module):
    """
    RepVGG Conv BN Relu Block, see https://arxiv.org/abs/2101.03697
    named RepVGGBlock in YOLOv6
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 act='relu',
                 deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.stride = stride  # not always 1

        self.act = nn.ReLU()  # get_activation(act)

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(in_channels)
                                 if out_channels == in_channels and stride == 1
                                 else None)
            self.rbr_dense = nn.Sequential(*[
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=groups,
                    bias=False),
                nn.BatchNorm2d(out_channels),
            ])
            self.rbr_1x1 = nn.Sequential(*[
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride,
                    padding_11,
                    groups=groups,
                    bias=False),
                nn.BatchNorm2d(out_channels),
            ])

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            x = self.rbr_reparam(inputs)
            y = self.act(x)
            return y

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        x = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        y = self.act(x)
        return y

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1]._mean
            running_var = branch[1]._variance
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1]._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros([self.in_channels, input_dim, 3, 3])

                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    def convert_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            self.stride,
            padding=1,
            groups=self.groups,
            bias=True)
        self.rbr_reparam.weight.set_value(kernel)
        self.rbr_reparam.bias.set_value(bias)
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class RepLayer(nn.Module):
    """
    RepLayer with RepConvs, like CSPLayer(C3) in YOLOv5/YOLOX
    named RepBlock in YOLOv6
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_repeats=1,
                 block=RepConv,
                 basic_block=RepConv):
        super(RepLayer, self).__init__()
        if block == RepConv:
            # in n/t/s
            self.conv1 = block(in_channels, out_channels)
            self.block = (nn.Sequential(*(block(out_channels, out_channels)
                                          for _ in range(num_repeats - 1)))
                          if num_repeats > 1 else None)

        elif block == BottleRep:
            # in m/l
            self.conv1 = BottleRep(
                in_channels, out_channels, basic_block=basic_block, alpha=True)
            num_repeats = num_repeats // 2
            self.block = nn.Sequential(*(BottleRep(
                out_channels, out_channels, basic_block=basic_block, alpha=True
            ) for _ in range(num_repeats - 1))) if num_repeats > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BottleRep(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 basic_block=RepConv,
                 alpha=False):
        super(BottleRep, self).__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if alpha:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class BepC3Layer(nn.Module):
    '''Beer-mug RepC3 Block
       named BepC3 in YOLOv6
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_repeats=1,
                 csp_e=0.5,
                 block=RepConv,
                 act='silu'):
        super(BepC3Layer, self).__init__()
        c_ = int(out_channels * csp_e)  # hidden channels
        self.cv1 = BaseConv(in_channels, c_, 1, 1, act="relu")
        self.cv2 = BaseConv(in_channels, c_, 1, 1, act="relu")
        self.cv3 = BaseConv(2 * c_, out_channels, 1, 1, act="relu")
        if act == 'silu':
            self.cv1 = BaseConv(in_channels, c_, 1, 1)
            self.cv2 = BaseConv(in_channels, c_, 1, 1)
            self.cv3 = BaseConv(2 * c_, out_channels, 1, 1)
        self.m = RepLayer(
            c_, c_, num_repeats, block=BottleRep, basic_block=block)

    def forward(self, x):
        return self.cv3(torch.concat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SimConv(nn.Module):
    """Simplified Conv BN ReLU"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super(SimConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimSPPF(nn.Module):
    """Simplified SPPF with SimConv, use relu"""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SimSPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = SimConv(in_channels, hidden_channels, 1, 1)
        self.mp = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = SimConv(hidden_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        concats = torch.concat([x, y1, y2, y3], 1)
        return self.conv2(concats)


class SPPF(nn.Module):
    """SPPF with BaseConv, use silu"""

    def __init__(self, in_channels, out_channels, kernel_size=5, act='silu'):
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, act=act)
        self.conv2 = BaseConv(
            hidden_channels * 4, out_channels, ksize=1, stride=1, act=act)
        self.mp = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        concats = torch.concat([x, y1, y2, y3], 1)
        return self.conv2(concats)


class SimCSPSPPF(nn.Module):
    """Simplified CSP SPPF with SimConv, use relu, YOLOv6 v3.0 added"""

    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5):
        super(SimCSPSPPF, self).__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(in_channels, c_, 1, 1)
        self.cv3 = SimConv(c_, c_, 3, 1)
        self.cv4 = SimConv(c_, c_, 1, 1)

        self.mp = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = SimConv(c_, c_, 3, 1)
        self.cv7 = SimConv(2 * c_, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        y1 = self.mp(x1)
        y2 = self.mp(y1)
        y3 = self.cv6(self.cv5(torch.concat([x1, y1, y2, self.mp(y2)], 1)))
        return self.cv7(torch.concat([y0, y3], 1))


class CSPSPPF(nn.Module):
    """CSP SPPF with BaseConv, use silu, YOLOv6 v3.0 added"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 e=0.5,
                 act='silu'):
        super(CSPSPPF, self).__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = BaseConv(in_channels, c_, 1, 1)
        self.cv2 = BaseConv(in_channels, c_, 1, 1)
        self.cv3 = BaseConv(c_, c_, 3, 1)
        self.cv4 = BaseConv(c_, c_, 1, 1)

        self.mp = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = BaseConv(4 * c_, c_, 1, 1)
        self.cv6 = BaseConv(c_, c_, 3, 1)
        self.cv7 = BaseConv(2 * c_, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.conv1(x)))
        y0 = self.cv2(x)
        y1 = self.mp(x1)
        y2 = self.mp(y1)
        y3 = self.cv6(self.cv5(torch.concat([x1, y1, y2, self.mp(y2)], 1)))
        return self.cv7(torch.concat([y0, y3], 1))


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True)

    def forward(self, x):
        return self.upsample_transpose(x)


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


@register
@serializable
class EfficientRep(nn.Module):
    """EfficientRep backbone of YOLOv6 n/s """
    __shared__ = ['width_mult', 'depth_mult', 'act', 'training_mode']

    # num_repeats, channels_list, 'P6' means add P6 layer
    arch_settings = {
        'P5': [[1, 6, 12, 18, 6], [64, 128, 256, 512, 1024]],
        'P6': [[1, 6, 12, 18, 6, 6], [64, 128, 256, 512, 768, 1024]],
    }

    def __init__(
            self,
            arch='P5',
            width_mult=0.33,
            depth_mult=0.50,
            return_idx=[2, 3, 4],
            training_mode='repvgg',
            fuse_P2=True,  # add P2 and return 4 layers
            cspsppf=True,
            act='relu'):
        super(EfficientRep, self).__init__()
        num_repeats, channels_list = self.arch_settings[arch]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]
        channels_list = [
            make_divisible(i * width_mult, 8) for i in (channels_list)
        ]
        self.return_idx = return_idx
        self.fuse_P2 = fuse_P2
        if self.fuse_P2:
            # stem,p2,p3,p4,p5: [0,1,2,3,4]
            self.return_idx = [1] + self.return_idx
        self._out_channels = [channels_list[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

        block = get_block(training_mode)
        # default block is RepConv
        self.stem = block(3, channels_list[0], 3, 2)
        self.blocks = []
        for i, (out_ch,
                num_repeat) in enumerate(zip(channels_list, num_repeats)):
            if i == 0: continue
            in_ch = channels_list[i - 1]
            stage = []

            repconv = block(in_ch, out_ch, 3, 2)
            self.add_module('stage{}_repconv'.format(i + 1), repconv)
            stage.append(repconv)

            replayer = RepLayer(out_ch, out_ch, num_repeat, block=block)
            self.add_module('stage{}_replayer'.format(i + 1), replayer)
            stage.append(replayer)

            if i == len(channels_list) - 1:
                if cspsppf:
                    simsppf_layer = SimCSPSPPF(out_ch, out_ch, kernel_size=5)
                    self.add_module('stage{}_simcspsppf'.format(i + 1), simsppf_layer)
                    stage.append(simsppf_layer)
                else:
                    simsppf_layer = SimSPPF(out_ch, out_ch, kernel_size=5)
                    self.add_module('stage{}_simsppf'.format(i + 1), simsppf_layer)
                    stage.append(simsppf_layer)
            self.blocks.append(nn.Sequential(*stage))

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i + 1 in self.return_idx:
                outputs.append(x)
        return outputs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]


@register
@serializable
class CSPBepBackbone(nn.Module):
    """CSPBepBackbone of YOLOv6 m/l """
    __shared__ = ['width_mult', 'depth_mult', 'act', 'training_mode']

    # num_repeats, channels_list, 'P6' means add P6 layer
    arch_settings = {
        'P5': [[1, 6, 12, 18, 6], [64, 128, 256, 512, 1024]],
        'P6': [[1, 6, 12, 18, 6, 6], [64, 128, 256, 512, 768, 1024]],
    }

    def __init__(self,
                 arch='P5',
                 width_mult=1.0,
                 depth_mult=1.0,
                 return_idx=[2, 3, 4],
                 csp_e=0.5,
                 training_mode='repvgg',
                 fuse_P2=True,
                 cspsppf=False,
                 act='relu'):
        super(CSPBepBackbone, self).__init__()
        num_repeats, channels_list = self.arch_settings[arch]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]
        channels_list = [
            make_divisible(i * width_mult, 8) for i in (channels_list)
        ]
        self.return_idx = return_idx
        self.fuse_P2 = fuse_P2
        if self.fuse_P2:
            # stem,p2,p3,p4,p5: [0,1,2,3,4]
            self.return_idx = [1] + self.return_idx
        self._out_channels = [channels_list[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

        block = get_block(training_mode)
        # RepConv(or RepVGGBlock) in M, but ConvBNSiLUBlock(or ConvWrapper) in L

        self.stem = block(3, channels_list[0], 3, 2)
        self.blocks = []
        if csp_e == 0.67:
            csp_e = float(2) / 3
        for i, (out_ch,
                num_repeat) in enumerate(zip(channels_list, num_repeats)):
            if i == 0: continue
            in_ch = channels_list[i - 1]
            stage = []

            repconv = block(in_ch, out_ch, 3, 2)
            self.add_module('stage{}_repconv'.format(i + 1), repconv)
            stage.append(repconv)

            bepc3layer = BepC3Layer(out_ch,
                                    out_ch,
                                    num_repeat,
                                    csp_e=csp_e,
                                    block=block,
                                    act=act)
            self.add_module('stage{}_bepc3layer'.format(i + 1), bepc3layer)
            stage.append(bepc3layer)

            if i == len(channels_list) - 1:
                if cspsppf:
                    # m/l never use cspsppf=True
                    if training_mode == 'conv_silu':
                        sppf_layer = CSPSPPF(out_ch, out_ch, kernel_size=5, act='silu')
                        self.add_module('stage{}_cspsppf'.format(i + 1),sppf_layer)
                        stage.append(sppf_layer)
                    else:
                        simsppf_layer = SimCSPSPPF(out_ch, out_ch, kernel_size=5)
                        self.add_module('stage{}_simcspsppf'.format(i + 1),simsppf_layer)
                        stage.append(simsppf_layer)
                else:
                    if training_mode == 'conv_silu':
                        sppf_layer = SPPF(out_ch, out_ch, kernel_size=5, act='silu')
                        self.add_module('stage{}_sppf'.format(i + 1), sppf_layer)
                        stage.append(sppf_layer)
                    else:
                        simsppf_layer = SimSPPF(out_ch, out_ch, kernel_size=5)
                        self.add_module('stage{}_simsppf'.format(i + 1), simsppf_layer)
                        stage.append(simsppf_layer)
            self.blocks.append(nn.Sequential(*stage))

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i + 1 in self.return_idx:
                outputs.append(x)
        return outputs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]


def get_block(mode):
    if mode == 'repvgg':
        return RepConv
    elif mode == 'conv_silu':
        return ConvBNSiLUBlock
    elif mode == 'conv_relu':
        return ConvBNReLUBlock
    else:
        raise ValueError('Unsupported mode :{}'.format(mode))


class ConvBNSiLUBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.base_block = BaseConv(in_channels, out_channels, kernel_size,
                                   stride, groups, bias)

    def forward(self, x):
        return self.base_block(x)


class ConvBNReLUBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.base_block = SimConv(in_channels, out_channels, kernel_size,
                                  stride, groups, bias)

    def forward(self, x):
        return self.base_block(x)


######################### YOLOv6 lite #########################


class ConvBN(nn.Module):
    '''Conv and BN without activation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=False):
        super().__init__()
        self.base_block = BaseConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            bias,
            act=None)

    def forward(self, x):
        return self.base_block(x)


class ConvBNHS(nn.Module):
    '''Conv and BN with Hardswish activation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 groups=1,
                 bias=False):
        super().__init__()
        self.base_block = BaseConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            bias,
            act='hardswish')

    def forward(self, x):
        return self.base_block(x)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out


class DPBlock(nn.Module):
    def __init__(self, in_channel=96, out_channel=96, kernel_size=3, stride=1):
        super().__init__()
        self.conv_dw_1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=out_channel,
            padding=(kernel_size - 1) // 2,
            stride=stride)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.act_1 = nn.Hardswish()
        self.conv_pw_1 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            groups=1,
            padding=0)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.act_2 = nn.Hardswish()

    def forward(self, x):
        x = self.act_1(self.bn_1(self.conv_dw_1(x)))
        x = self.act_2(self.bn_2(self.conv_pw_1(x)))
        return x

    def forward_fuse(self, x):
        x = self.act_1(self.conv_dw_1(x))
        x = self.act_2(self.conv_pw_1(x))
        return x


class DarknetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv_1 = ConvBNHS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv_2 = DPBlock(
            in_channel=hidden_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out


class CSPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expand_ratio=0.5):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.conv_1 = ConvBNHS(in_channels, mid_channels, 1, 1, 0)
        self.conv_2 = ConvBNHS(in_channels, mid_channels, 1, 1, 0)
        self.conv_3 = ConvBNHS(2 * mid_channels, out_channels, 1, 1, 0)
        self.blocks = DarknetBlock(mid_channels, mid_channels, kernel_size, 1.0)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.blocks(x_1)
        x_2 = self.conv_2(x)
        x = torch.concat((x_1, x_2), dim=1)
        x = self.conv_3(x)
        return x


def channel_shuffle(x, groups):
    _, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    # reshape
    x = x.reshape([-1, groups, channels_per_group, height, width])
    x = x.transpose([0, 2, 1, 3, 4])
    # flatten
    x = x.reshape([-1, groups * channels_per_group, height, width])
    return x


class Lite_EffiBlockS1(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride):
        super().__init__()
        self.conv_pw_1 = ConvBNHS(
            in_channels=in_channels // 2,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        self.conv_dw_1 = ConvBN(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            groups=mid_channels)
        self.se = SEBlock(mid_channels)
        self.conv_1 = ConvBNHS(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)

    def forward(self, inputs):
        x1, x2 = torch.split(
            inputs,
            num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            axis=1)
        x2 = self.conv_pw_1(x2)
        x3 = self.conv_dw_1(x2)
        x3 = self.se(x3)
        x3 = self.conv_1(x3)
        out = torch.concat([x1, x3], axis=1)
        return channel_shuffle(out, 2)


class Lite_EffiBlockS2(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride):
        super().__init__()
        # branch1
        self.conv_dw_1 = ConvBN(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels)
        self.conv_1 = ConvBNHS(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        # branch2
        self.conv_pw_2 = ConvBNHS(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        self.conv_dw_2 = ConvBN(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            groups=mid_channels // 2)
        self.se = SEBlock(mid_channels // 2)
        self.conv_2 = ConvBNHS(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        self.conv_dw_3 = ConvBNHS(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels)
        self.conv_pw_3 = ConvBNHS(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)

    def forward(self, inputs):
        x1 = self.conv_dw_1(inputs)
        x1 = self.conv_1(x1)
        x2 = self.conv_pw_2(inputs)
        x2 = self.conv_dw_2(x2)
        x2 = self.se(x2)
        x2 = self.conv_2(x2)
        out = torch.concat([x1, x2], axis=1)
        out = self.conv_dw_3(out)
        out = self.conv_pw_3(out)
        return out


def make_divisible_lite(v, divisor=16):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@register
@serializable
class Lite_EffiBackbone(nn.Module):
    """Lite_EffiBackbone of YOLOv6-lite"""
    __shared__ = ['width_mult']

    def __init__(self,
                 width_mult=1.0,
                 return_idx=[2, 3, 4],
                 out_channels=[24, 32, 64, 128, 256],
                 num_repeat=[1, 3, 7, 3],
                 scale_size=0.5):
        super().__init__()
        self.return_idx = return_idx
        out_channels = [
            make_divisible_lite(i * width_mult) for i in out_channels
        ]
        mid_channels = [
            make_divisible_lite(
                int(i * scale_size), divisor=8) for i in out_channels
        ]

        out_channels[0] = 24
        self.conv_0 = ConvBNHS(
            in_channels=3,
            out_channels=out_channels[0],
            kernel_size=3,
            stride=2,
            padding=1)

        self.lite_effiblock_1 = self.build_block(
            num_repeat[0], out_channels[0], mid_channels[1], out_channels[1])

        self.lite_effiblock_2 = self.build_block(
            num_repeat[1], out_channels[1], mid_channels[2], out_channels[2])

        self.lite_effiblock_3 = self.build_block(
            num_repeat[2], out_channels[2], mid_channels[3], out_channels[3])

        self.lite_effiblock_4 = self.build_block(
            num_repeat[3], out_channels[3], mid_channels[4], out_channels[4])

        self._out_channels = [out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.conv_0(x)
        x = self.lite_effiblock_1(x)
        x = self.lite_effiblock_2(x)
        outputs.append(x)
        x = self.lite_effiblock_3(x)
        outputs.append(x)
        x = self.lite_effiblock_4(x)
        outputs.append(x)
        return outputs

    @staticmethod
    def build_block(num_repeat, in_channels, mid_channels, out_channels):
        block_list = nn.Sequential()
        for i in range(num_repeat):
            if i == 0:
                block = Lite_EffiBlockS2(
                    in_channels=in_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=2)
            else:
                block = Lite_EffiBlockS1(
                    in_channels=out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=1)
            block_list.add_sublayer(str(i), block)
        return block_list

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]
