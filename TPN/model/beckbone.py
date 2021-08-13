#Res50_I3D
#slow 模式，将最后两层特征图输出

# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = ["ResNet_SlowFast"]

class ResNet_SlowFast():
    def __init__(self, layers=50, is_training=True):
        self.layers = layers
    
        self.is_training = is_training

    def net(self, input, data_format="NCDHW"):
        layers = self.layers
        supported_layers = [ 50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        channels = 3
        seglen = int(input.shape[-3]/channels)
        segnum = input.shape[1]
        short_size = input.shape[3]
        input = fluid.layers.reshape(
            x=input, shape=[-1, seglen, channels, short_size, short_size])
        input = fluid.layers.transpose(x=input, perm=[0, 2, 1, 3, 4])


        if  layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]
        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=[1, 7, 7],
            stride=[1, 2, 2],
            padding = [0,3,3],
            act='relu',
            name="conv1",
            data_format=data_format)
        conv = fluid.layers.pool3d(
            input=conv,
            pool_size=[1,3,3],
            pool_stride=[1,2,2],
            pool_padding=[0,1,1],
            pool_type='max',
            data_format=data_format)
        out=[]
        for block in range(len(depth)):
            for i in range(depth[block]):
                if layers in [101, 152] and block == 2:
                    if i == 0:
                        conv_name = "res" + str(block + 2) + "a"
                    else:
                        conv_name = "res" + str(block + 2) + "b" + str(i)
                else:
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                if(block<2):
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        name=conv_name,
                        data_format=data_format,
                        type = '2d')
                else:
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        name=conv_name,
                        data_format=data_format,
                        type='3d')
            if (block > 1):
                out.append(conv)
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      padding='SAME',
                      name=None,
                      data_format='NCDHW'):
        assert data_format == 'NCDHW'
        if (isinstance(filter_size, int)):
            filter_size_3d = [filter_size] * 3
        else:
            filter_size_3d = filter_size
        if (isinstance(stride, int)):
            stride_3d = [stride] * 3
        else:
            stride_3d = stride
        conv = fluid.layers.conv3d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size_3d,
            stride=stride_3d,
            padding=padding,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1',
            data_format=data_format)

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            is_test=(not self.is_training),
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def shortcut(self, input, ch_out, stride, is_first, name, data_format):
        if data_format == 'NCDHW':
            ch_in = input.shape[1]
        else:
            ch_in = input.shape[-1]
        if ch_in != ch_out or stride != [1,1,1] or is_first == True:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name, data_format=data_format)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name, data_format,type):
        if type=='2d':
            conv0 = self.conv_bn_layer(
                input=input,
                num_filters=num_filters,
                filter_size=1,
                act='relu',
                name=name + "_branch2a",
                data_format=data_format)
        else:
            conv0 = self.conv_bn_layer(
                input=input,
                num_filters=num_filters,
                filter_size=[3,1,1],
                act='relu',
                name=name + "_branch2a",
                data_format=data_format)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=[1,3,3],
            stride=[1,stride,stride],
            padding=[0,1,1],
            act='relu',
            name=name + "_branch2b",
            data_format=data_format)

        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
            data_format=data_format)

        short = self.shortcut(
            input,
            num_filters * 4,
            [1,stride,stride],
            is_first=False,
            name=name + "_branch1",
            data_format=data_format)

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")







