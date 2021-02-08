import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import BatchNorm, Conv3D

class ConvBNLayer_3d(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer_3d, self).__init__(name_scope)

        self._conv = Conv3D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock_3d(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock_3d, self).__init__(name_scope)

        self.conv0 = ConvBNLayer_3d(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            act='relu')
        self.conv1 = ConvBNLayer_3d(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')

        if not shortcut:
            self.short = ConvBNLayer_3d(
                self.full_name(),
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=3,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv1)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)

class ResNet3D(fluid.dygraph.Layer):
    def __init__(self, name_scope, channels):
        super(ResNet3D, self).__init__(name_scope)

        self.channels = channels


###### begin of 3D network  #### 
        
        depth_3d = [2, 2, 2]  #part of 3dresnet18
        num_filters_3d = [128, 256, 512]

        self.bottleneck_block_list_3d = []
        num_channels_3d = self.channels
        for block in range(len(depth_3d)):
            shortcut = False
            for i in range(depth_3d[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock_3d(
                        self.full_name(),
                        num_channels=num_channels_3d,
                        num_filters=num_filters_3d[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                num_channels_3d = bottleneck_block._num_channels_out
                self.bottleneck_block_list_3d.append(bottleneck_block)
                shortcut = True       

#### end of 3D network  #### 
 

    def forward(self, inputs, label=None):
  
        y = inputs

        for bottleneck_block in self.bottleneck_block_list_3d:
            y = bottleneck_block(y)
           

        y = fluid.layers.pool3d(input=y, pool_size=7, pool_type='avg', global_pooling=True)
        

        return y


if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = ResNet3D('resnet', channels=10)
        img = np.zeros([1, 10, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img).numpy()
        print(outs)