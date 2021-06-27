# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

__all__ = ["DLA60", "DLA60x", "DLA60x_c", "DLA46x_c", "DLA46_c", "DLA34"]


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 padding=0,
                 dilation=1,
                 name=None,
                 data_format="NCHW"):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding, #(filter_size - 1) // 2,
            dilation=dilation,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            data_format=data_format)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
            data_layout=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class DlaBottleneckBlock(nn.Layer):
    expansion = 2

    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dilation=1, 
                 cardinality=1, 
                 base_width=64,
                 name=None,
                 data_format="NCHW"):
        super(DlaBottleneckBlock, self).__init__()
        mid_planes = int(math.floor(num_filters * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=mid_planes,
            filter_size=1,
            act="relu",
            name=name + "_conv0",
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            num_channels=mid_planes,
            num_filters=mid_planes,
            filter_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            act="relu",
            name=name + "_conv1",
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            num_channels=mid_planes,
            num_filters=num_filters,
            filter_size=1,
            act=None,
            name=name + "_conv2",
            data_format=data_format)

    def forward(self, inputs, residual=None):
        if residual is None:
            residual = inputs

        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        y = paddle.add(x=residual, y=conv2)
        y = F.relu(y)
        return y


class DlaBasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dilation=1,
                 name=None,
                 data_format="NCHW",
                 **_):
        super(DlaBasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            act="relu",
            name=name + "_conv0",
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            padding=dilation,
            dilation=dilation,
            act=None,
            name=name + "_conv1",
            data_format=data_format)


    def forward(self, inputs, residual=None):
        if residual is None:
            residual = inputs

        y = self.conv0(inputs)
        conv1 = self.conv1(y)


        y = paddle.add(x=residual, y=conv1)
        y = F.relu(y)
        return y

class DlaRoot(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, residual, name=None, data_format="NCHW"):
        super(DlaRoot, self).__init__()
        self.conv0 = ConvBNLayer(
            num_channels=in_channels,
            num_filters=out_channels,
            filter_size=1,
            padding=(kernel_size - 1) // 2,
            act=None,
            name=name + "_conv0",
            data_format=data_format)

        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv0(paddle.concat(x=x, axis=1))
        if self.residual:
            x = paddle.add(x=x, y=children[0])
        x = F.relu(x)

        return x

class DlaTree(nn.Layer):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 dilation=1, cardinality=1, base_width=64, name=None,
                 level_root=False, root_dim=0, root_kernel_size=1, root_residual=False,
                 data_format="NCHW"):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample = MaxPool2D(stride, stride=stride) if stride > 1 else None
        self.project = None
        cargs = dict(dilation=dilation, cardinality=cardinality, base_width=base_width, data_format=data_format)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, name=name + '_tree1', **cargs)
            self.tree2 = block(out_channels, out_channels, 1, name=name + '_tree2', **cargs)
            if in_channels != out_channels:
                # NOTE the official impl/weights have  project layers in levels > 1 case that are never
                # used, I've moved the project layer here to avoid wasted params but old checkpoints will
                # need strict=False while loading.
                self.project = ConvBNLayer(
                    num_channels=in_channels,
                    num_filters=out_channels,
                    filter_size=1,
                    act=None,
                    name=name + "_project",
                    data_format=data_format)
        else:
            cargs.update(dict(root_kernel_size=root_kernel_size, root_residual=root_residual))
            self.tree1 = DlaTree(
                levels - 1, block, in_channels, out_channels, stride, root_dim=0, name=name + '_tree1', **cargs)
            self.tree2 = DlaTree(
                levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, name=name + '_tree2', **cargs)
        if levels == 1:
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size, root_residual, name=name + '_root', data_format=data_format)
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample is not None else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(nn.Layer):
    def __init__(self, levels, channels, class_dim=1000, input_image_channel=3, data_format="NCHW",
                cardinality=1, base_width=64, block=DlaBottleneckBlock, residual_root=False, drop_rate=0.0):
        super(DLA, self).__init__()

        self.levels = levels
        self.channels = channels
        self.data_format = data_format
        self.input_image_channel = input_image_channel
        self.cardinality = cardinality
        self.base_width = base_width
        self.residual_root = residual_root
        self.drop_rate = drop_rate

        self.base_layer = ConvBNLayer(
            num_channels=self.input_image_channel,
            num_filters=self.channels[0],
            filter_size=7,
            stride=1,
            act="relu",
            padding=3,
            name="base_layer",
            data_format=self.data_format)
        self.level0 = self._make_conv_level(self.channels[0], self.channels[0], self.levels[0], name='level0')
        self.level1 = self._make_conv_level(self.channels[0], self.channels[1], self.levels[1], stride=2, name='level1')
        cargs = dict(cardinality=cardinality, base_width=base_width, root_residual=residual_root, data_format=data_format)
        self.level2 = DlaTree(levels[2], block, channels[1], channels[2], 2, level_root=False, name='level2', **cargs)
        self.level3 = DlaTree(levels[3], block, channels[2], channels[3], 2, level_root=True, name='level3', **cargs)
        self.level4 = DlaTree(levels[4], block, channels[3], channels[4], 2, level_root=True, name='level4', **cargs)
        self.level5 = DlaTree(levels[5], block, channels[4], channels[5], 2, level_root=True, name='level5', **cargs)

        self.num_features = channels[-1]
        self.global_pool = AdaptiveAvgPool2D(1, data_format=self.data_format)
        stdv = 1.0 / math.sqrt(self.num_features * 1.0)
        self.fc = Linear(
            self.num_features,
            class_dim,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name="fc_0.w_0"),
            bias_attr=ParamAttr(name="fc_0.b_0"))

        self.drop = Dropout(p=self.drop_rate)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1, name=None):
        modules = nn.LayerList()
        for i in range(convs):
            modules.append(ConvBNLayer(
                num_channels=inplanes,
                num_filters=planes,
                filter_size=3,
                stride=stride if i == 0 else 1,
                act="relu",
                padding=dilation,
                dilation=dilation,
                name=name + '_' + str(i),
                data_format=self.data_format))
            inplanes = planes
        return modules

    def forward_features(self, x):
        x = self.base_layer(x)
        for block in self.level0:
            x = block(x)
        for block in self.level1:
            x = block(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)
        return x

    def forward(self, inputs):
        x = self.forward_features(inputs)
        x = self.global_pool(x)
        x = paddle.reshape(x, shape=[-1, self.num_features])
        if self.drop_rate > 0.:
            x = self.drop(x)
        x = self.fc(x)
        return x

def DLA60(**args):
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 3, 1], channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneckBlock, **args)
    model = DLA(**model_kwargs)
    return model

def DLA60x(**args):
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 3, 1], channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneckBlock, cardinality=32, base_width=4, **args)
    model = DLA(**model_kwargs)
    return model

def DLA60x_c(**args):
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 3, 1], channels=[16, 32, 64, 64, 128, 256],
        block=DlaBottleneckBlock, cardinality=32, base_width=4, **args)
    model = DLA(**model_kwargs)
    return model

def DLA34(**args):
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 128, 256, 512],
        block=DlaBasicBlock, **args)
    model = DLA(**model_kwargs)
    return model

def DLA46_c(**args):
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 64, 128, 256],
        block=DlaBottleneckBlock, **args)
    model = DLA(**model_kwargs)
    return model

def DLA46x_c(**args):
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 64, 128, 256],
        block=DlaBottleneckBlock, cardinality=32, base_width=4, **args)
    model = DLA(**model_kwargs)
    return model
