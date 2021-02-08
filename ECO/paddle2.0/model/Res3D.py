import paddle
from paddle.nn import BatchNorm3D, Conv3D

class ConvBNLayer_3d(paddle.nn.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1):
        super(ConvBNLayer_3d, self).__init__(name_scope)

        self._conv = Conv3D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        self._batch_norm = BatchNorm3D(num_filters)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = paddle.nn.functional.relu(y)
        y = self._batch_norm(y)
        y = paddle.nn.functional.relu(y)
        return y


class BottleneckBlock_3d(paddle.nn.Layer):
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
            filter_size=3)
        self.conv1 = ConvBNLayer_3d(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride)

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

        y = paddle.add(x=short, y=conv1)

        y = paddle.nn.functional.relu(y)

        return y

class ResNet3D(paddle.nn.Layer):
 
    def __init__(self, name_scope, channels, modality="RGB"):
        super(ResNet3D, self).__init__(name_scope)

        self.modality = modality
        self.channels = channels


###### begin of 3D network
        
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


#### end of 3D network
 

    def forward(self, inputs, label=None):
  
        y = inputs

        for bottleneck_block in self.bottleneck_block_list_3d:
            y = bottleneck_block(y)
           
        y = paddle.nn.functional.adaptive_avg_pool3d(y, 1)

        return y


if __name__ == '__main__':
    network = ResNet3D('resnet', modality='RGB', channels=10)
    img = paddle.zeros([1, 10, 3, 224, 224])
    outs = network(img).numpy()
    print(outs.shape)