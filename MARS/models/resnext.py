#coding=utf-8
from __future__ import division
import math
from functools import partial
import pdb

import paddle
import paddle.fluid as fluid

__all__ = ['ResNeXt', 'resnet101', 'resnet152']

def downsample_basic_block(x, planes, stride):
    out = fluid.layers.pool3d(x, kernel_size=1, stride=stride)
    zero_pads = fluid.layers.zeros(shape=[out.shape[0], planes - out.shape[1], out.shape[2], out.shape[3],out.shape[4]])
    out = fluid.dygraph.base.to_variable(fluid.layers.concat([out.data, zero_pads], axis=1))
    return out


class ConvBNLayer(fluid.dygraph.Layer):
    """
    由于卷积、批归一化一般连在一起使用，所以单独的将其写在一起，可以作为一个整体使用
    """
    def __init__(self,num_channels,num_filters,filter_size,stride=1,groups=1,param_attr=None,act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = fluid.dygraph.Conv3D(num_channels,num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm = fluid.dygraph.BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        """
        将输入前向传播，计算结果
        """
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class ResNeXtBottleneck(fluid.dygraph.Layer):
    expansion = 2
    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.MSRAInitializer(uniform=True))
        self.conv1 = ConvBNLayer(num_channels=inplanes,num_filters=mid_planes,filter_size=1, param_attr = param_attr,act="relu")
        self.conv2 = ConvBNLayer(num_channels=mid_planes,num_filters=mid_planes,filter_size=3,
                    stride=stride,groups=cardinality, param_attr = param_attr,act="relu")
        self.conv3 = ConvBNLayer(num_channels=mid_planes,num_filters=planes * self.expansion,filter_size=1, 
                    param_attr = param_attr,act=None)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input, label=None):
        """
        将输入前向传播，计算结果
        """
        residual = input
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out = fluid.layers.elementwise_add(x=out, y=residual, act='relu')
        return out


class ResNeXt(fluid.dygraph.Layer):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 input_channels=3,
                 output_layers=[]):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.MSRAInitializer(uniform=True))
        self.conv1 = ConvBNLayer(num_channels=input_channels,num_filters=64,filter_size=7,
                    stride=(1, 2, 2), param_attr = param_attr,act="relu")
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer( block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        self.last_duration = int(math.ceil(sample_duration / 16))
        self.last_size = int(math.ceil(sample_size / 32))
        self.fc = fluid.dygraph.Linear(cardinality * 32 * block.expansion, num_classes,
                                param_attr=fluid.ParamAttr(initializer=fluid.initializer.MSRAInitializer(uniform=True)), 
                                bias_attr=paddle.fluid.ParamAttr(initializer=None),act="softmax")
        self.output_layers = output_layers
        self.lastfeature_size=cardinality * 32 * block.expansion


    def _make_layer(self,block,planes,blocks,shortcut_type,cardinality,stride=1):
        """
        根据给定参数构建残差块
        参数：
        Return：
            构建的网络层序列
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = fluid.dygraph.Sequential(
                    fluid.dygraph.Conv3D(self.inplanes,planes * block.expansion,filter_size=1,stride=stride, 
                    param_attr = fluid.ParamAttr(initializer=fluid.initializer.MSRAInitializer(uniform=True)),
                    bias_attr = False,act=None), fluid.dygraph.BatchNorm(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return fluid.dygraph.Sequential(*layers)

    def forward(self, x):
        """
        将输入前向传播，计算结果
        """
        x = self.conv1(x)
        x = fluid.layers.pool3d(x, pool_size=(3, 3, 3),pool_type='max',pool_stride=2, pool_padding=1)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = fluid.layers.pool3d(x4, pool_size=(self.last_duration, self.last_size, self.last_size),pool_type='avg',pool_stride=1)
        x6 = fluid.layers.reshape(x5,[x5.shape[0], -1])
        x7 = self.fc(x6)
        if len(self.output_layers) == 0:
            return x7
        else:
            out = []
            out.append(x7)
            for i in self.output_layers:
                if i == 'avgpool':
                    out.append(x6)
                if i == 'layer4':
                    out.append(x4)
                if i == 'layer3':
                    out.append(x3)

        return out

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()
    # pdb.set_trace()
    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    print("Layers to finetune : ", ft_module_names)

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append(v)
                break
        # else:
        #     parameters.append({'params': v, 'lr': 0.0})
    #pdb.set_trace()
    return parameters

def resnet50(**kwargs):
    """
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """
    101层的resnet作为主干网络
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """
    152层的resnet作为主干网络
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


