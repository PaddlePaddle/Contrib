#encoding=utf8
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

import paddle
import paddle.nn as nn
from paddle import Tensor

class Identity(nn.Layer):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        # >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        # >>> input = paddle.randn(128, 20)
        # >>> output = m(input)

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input

def fuse_conv_bn(conv, bn):
    """During inference, the functionary of batch norm layers is turned off but
    only the mean and var alone channels are used, which exposes the chance to
    fuse it with the preceding conv layers to save computations and simplify
    network structures."""
    conv_w = conv.weight

    conv_b = conv.bias if conv.bias is not None else paddle.zeros_like(
        bn._mean)

    factor = bn.weight / paddle.sqrt(bn._variance + bn._epsilon)
    p_temp = conv_w * paddle.reshape(factor,[conv._out_channels, 1, 1, 1])
    conv.weight = paddle.static.create_parameter(shape=p_temp.shape, dtype = p_temp.dtype)
    conv.weight.set_value(p_temp)
    p_temp = (conv_b - bn._mean) * factor + bn.bias
    conv.bias = paddle.static.create_parameter(shape=p_temp.shape, dtype =  p_temp.dtype)
    conv.bias.set_value(p_temp)
    return conv


def fuse_module(m):
    last_conv = None
    last_conv_name = None

    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2D, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = fuse_conv_bn(last_conv, child)
            m._sub_layers[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            
            m._sub_layers[name] = Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2D):
            last_conv = child
            last_conv_name = name
        else:
            fuse_module(child)
    return m
