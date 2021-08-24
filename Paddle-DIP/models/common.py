# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
from .downsampler import Downsampler


class Swish(nn.Layer):
    """
    https://arxiv.org/abs/1710.05941
    The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun='LeakyReLU'):
    """
    Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


class Conv(nn.Layer):
    def __init__(self, in_f, out_f, kernel_size, stride=1,
                 bias=True, act_fun='LeakyReLU',
                 pad='zero', downsample_mode='stride'):
        super(Conv, self).__init__()
        self.downsampler = None
        if stride != 1 and downsample_mode != 'stride':

            if downsample_mode == 'avg':
                self.downsampler = nn.AvgPool2D(stride, stride)
            elif downsample_mode == 'max':
                self.downsampler = nn.MaxPool2D(stride, stride)
            elif downsample_mode in ['lanczos2', 'lanczos3']:
                self.downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5,
                                               preserve_size=True)

        self.padder = None
        to_pad = int((kernel_size - 1) / 2)
        if pad == 'reflection':
            self.padder = nn.Pad2D(to_pad, mode='replicate')
            to_pad = 0

        self.conv = nn.Conv2D(in_f, out_f, kernel_size, stride, padding=to_pad)

        self.bn = nn.BatchNorm2D(out_f)
        self.act = act(act_fun)

    def forward(self, x):
        if self.padder:
            x = self.padder(x)

        x = self.conv(x)

        if self.downsampler:
            x = self.downsampler(x)

        x = self.bn(x)
        x = self.act(x)

        return x
