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
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

__all__ = [
    "HarDNet68", "HarDNet85", "HarDNet68ds", "HarDNet39ds"
]

class CombConvLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False, name=None):
        super().__init__()
        self.layer1 = ConvBNLayer(in_channels, out_channels, kernel, name=name + '_layer1')
        self.layer2 = ConvBNLayer(out_channels, out_channels, 3, stride=stride, pad=1, groups=out_channels,
                        act=None, name=name + '_layer2')
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        return y

class HarDBlock(nn.Layer):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False, name=None):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = nn.LayerList()
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch, name=name + f'_layer_{i}'))
            else:
                layers_.append(ConvBNLayer(inch, outch, 3, pad=1, name=name + f'_layer_{i}'))
            
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = layers_
        
    def forward(self, x):
        layers_ = [x]
        
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:            
                x = paddle.concat(x=tin, axis=1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                (i == t-1) or (i%2 == 1):
                out_.append(layers_[i])
        out = paddle.concat(x=out_, axis=1)
        return out


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 act="relu6",
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=name + '_bn_scale'),
            bias_attr=ParamAttr(name + '_bn_offset'),
            moving_mean_name=name + '_bn_mean',
            moving_variance_name=name + '_bn_variance')

    def forward(self, input):
        y = self._conv(input)
        y = self._batch_norm(y)
        return y


class HarDNet(nn.Layer):
    def __init__(self, depth_wise=False, arch=85, dropout=0.1, class_dim=1000):
        super(HarDNet, self).__init__()
        first_ch  = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = dropout

        #HarDNet68
        ch_list = [  128, 256, 320, 640, 1024]
        gr       = [  14, 16, 20, 40,160]
        n_layers = [   8, 16, 16, 16,  4]
        downSamp = [   1,  0,  1,  1,  0]
        
        if arch==85:
            #HarDNet85
            first_ch  = [48, 96]
            ch_list = [  192, 256, 320, 480, 720, 1280]
            gr       = [  24,  24,  28,  36,  48, 256]
            n_layers = [   8,  16,  16,  16,  16,   4]
            downSamp = [   1,   0,   1,   0,   1,   0]
            drop_rate = 0.2
        elif arch==39:
            #HarDNet39
            first_ch  = [24, 48]
            ch_list = [  96, 320, 640, 1024]
            grmul = 1.6
            gr       = [  16,  20, 64, 160]
            n_layers = [   4,  16,  8,   4]
            downSamp = [   1,   1,  1,   0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        blks = len(n_layers)
        self.base = nn.LayerList()

        count = 0
        # First Layer: Standard Conv3x3, Stride=2
        # tmp_block = self.add_sublayer("base_" + str(count), ConvBNLayer(num_channels=3, num_filters=first_ch[0], filter_size=3,
        #                stride=2, name="base_" + str(count)))
        self.base.append(ConvBNLayer(num_channels=3, num_filters=first_ch[0], filter_size=3,
                       stride=2, pad=1, name="base_" + str(count)))
        count += 1

        # Second Layer
        self.base.append(ConvBNLayer(num_channels=first_ch[0], num_filters=first_ch[1], filter_size=second_kernel,
            pad=1, name="base_" + str(count)))
        count += 1

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(MaxPool2D(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(ConvBNLayer(num_channels=first_ch[1], num_filters=first_ch[1], filter_size=3,
                stride=1, pad=1, groups=first_ch[1], act=None, name="base_" + str(count)))
            count += 1

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise, name="base_" + str(count))
            count += 1
            ch = blk.get_out_ch()
            self.base.append(blk)
            
            if i == blks-1 and arch == 85:
                self.base.append(Dropout(p=0.1))
            
            self.base.append(ConvBNLayer(ch, ch_list[i], 1, name="base_" + str(count)))
            count += 1
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(MaxPool2D(kernel_size=2, stride=2))
                else:
                    self.base.append(ConvBNLayer(ch, ch, 3, stride=2, pad=1, groups=ch, 
                        act=None, name="base_" + str(count)))
                    count += 1

        ch = ch_list[blks-1]
        self.num_features = ch
        self.base.append(AdaptiveAvgPool2D(1))

        self.drop = Dropout(p=drop_rate)
        stdv = 1.0 / math.sqrt(ch * 1.0)
        self.fc = Linear(
            ch,
            class_dim,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name="fc_0.w_0"),
            bias_attr=ParamAttr(name="fc_0.b_0"))

    def forward(self, x):
        for layer in self.base:
            x = layer(x)

        x = paddle.reshape(x, shape=[-1, self.num_features])
        x = self.drop(x)
        x = self.fc(x)
        return x

def HarDNet68(**args):
    model = HarDNet(depth_wise=False, arch=68, **args)
    return model

def HarDNet85(**args):
    model = HarDNet(depth_wise=False, arch=85, **args)
    return model

def HarDNet68ds(**args):
    model = HarDNet(depth_wise=True, arch=68, **args)
    return model

def HarDNet39ds(**args):
    model = HarDNet(depth_wise=True, arch=39, **args)
    return model
