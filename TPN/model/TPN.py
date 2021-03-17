from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
import math
class TPN():
    def __init__(self, config,seg_num,is_training=True):
        self.config = config
        self.is_training = is_training
        self.seg_num = seg_num
    def net(self,input,label):

        out = self.SpatialModulation(input)

        out = self.TemporalModulation(out)

        out = self.level_fusion(out)

        out = self.pyramid_fusion(out)

        loss_aus= self.AuxHead(input[0],label)
        return out,loss_aus

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name='',
                      padding='SAME',
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
        is_training = self.is_training
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
            data_format=data_format)
        bn_name = name+'_bn'
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            is_test=(not is_training),
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance'
        )

    # 简化起见只支持两层金字塔
    def SpatialModulation(self,input, name=None, data_format='NCDHW'):
        layers_num = len(input)
        assert layers_num == 2
        layer_bottom, layer_top = input
        planes = layer_top.shape[1]
        # 将bottom层的特征与top层的特征在shape和channel维度上对齐
        bottom_out = self.conv_bn_layer(input=layer_bottom,
                                   num_filters=planes,
                                   filter_size=[1, 3, 3],
                                   padding=[0, 1, 1],
                                   stride=[1, 2, 2],
                                   act='relu',
                                   name = 'SpatialModulation',
                                   data_format=data_format)
        return [bottom_out, layer_top]

    # 简化起见只支持两层金字塔
    def TemporalModulation(self,input):
        layers_num = len(input)
        assert layers_num == 2
        down_scale = self.config['temporal_modulation_config']['down_scale']
        planes = self.config['TPN']['out_channels']

        outs = []
        for i in range(layers_num):
            out = input[i]
            out = fluid.layers.conv3d(input=out, num_filters=planes, filter_size=[3, 1, 1],
                                      stride=1, padding=[1, 0, 0], groups=32, bias_attr=False,
                                      param_attr=ParamAttr(name='TemporalModulation{}'.format(i) + "_weights"),)
            out = fluid.layers.pool3d(input=out, pool_size=[down_scale[i], 1, 1],pool_type='max',
                                      pool_stride=[down_scale[i], 1, 1], ceil_mode=True)
            outs.append(out)
        return outs

    # 时域上采样函数
    # def Upsamping(self,input, scale=2):
    #     out_shape = (int(input.shape[-3] * scale), input.shape[-2], input.shape[-1])
    #     return fluid.layers.interpolate(input=input, out_shape=out_shape, resample='TRILINEAR', data_format='NCDHW')

    # 时域上采样函数
    def Upsamping(self, input, scale=2):
        if(scale==1):
            output = input
        elif(scale==2):
            output = fluid.layers.expand(input,[1,1,2,1,1])
        else:
            output=None
        return output

    # 时域下采样函数
    def Downsamping(self,input, scale, position='after', kernel_size=(3, 1, 1), groups=1,
                    padding=(1, 0, 0), norm=False,planes=None,name='Downsamping'):
        if planes == None:
            planes = input.shape[1]
        if (position == 'befor'):
            input = fluid.layers.pool3d(input, pool_stride=[scale, 1, 1], pool_size=[scale, 1, 1],
                                        pool_type='max',ceil_mode=True)
        if (norm == False):
            input = fluid.layers.conv3d(input=input, num_filters=planes, filter_size=kernel_size, padding=padding,
                                        bias_attr=False,param_attr=ParamAttr(name=name+'_weights'), groups=groups)
        else:
            input = self.conv_bn_layer(input=input, num_filters=planes, filter_size=kernel_size, groups=groups,
                                  padding=padding, act='relu',name=name)
        if (position == 'after'):
            input = fluid.layers.pool3d(input, pool_stride=[scale, 1, 1], pool_size=[scale, 1, 1],
                                        pool_type='max',ceil_mode=True)
        return input

    # 简化起见只支持两层金字塔,采用并行的模式
    def level_fusion(self, input):
        layers_num = len(input)
        assert layers_num == 2
        down, top = input
        mid_channel = self.config['level_fusion']['mid_channels']
        out_channel = self.config['level_fusion']['out_channels']
        # 先计算top-town的信息融合，采用upsample进行对齐
        T_down = down.shape[-3]
        T_top = top.shape[-3]
        scale = int(T_down / T_top)

        top_upsample = self.Upsamping(input=top, scale=scale)

        down_out = fluid.layers.elementwise_add(down, top_upsample)

        # 将2层金字塔的信息融合成1层
        down_out1 = self.Downsamping(input=down_out, scale=scale, position='befor', padding=(0, 0, 0),
                               kernel_size=1, groups=32, norm=True,planes=mid_channel,
                               name='level_fusion_topdown_downout')
        top_out = self.Downsamping(input=top, scale=1, position='befor', padding=(0, 0, 0),
                              kernel_size=1, groups=32, norm=True,planes=mid_channel,
                              name='level_fusion_topdown_topout')
        top_down_out = fluid.layers.concat(input=[down_out1, top_out], axis=1)
        top_down_out = self.conv_bn_layer(input=top_down_out, num_filters=out_channel, filter_size=1,
                                     stride=1, act='relu',name='level_fusion_topdown')

        # 再计算down-top的信息融合，采用downsample进行对齐
        down = down_out
        down_downsamping = self.Downsamping(input=down, scale=scale, position='after')
        top_out = fluid.layers.elementwise_add(top, down_downsamping)

        # 将2层金字塔的信息融合成1层
        down_out2 = self.Downsamping(input=down, scale=scale, position='befor', padding=(0, 0, 0),
                               kernel_size=1, groups=32, norm=True,planes=mid_channel,
                               name='level_fusion_downtop_downout')
        top_out = self.Downsamping(input=top_out, scale=1, position='befor', padding=(0, 0, 0),
                              kernel_size=1, groups=32, norm=True,planes=mid_channel,
                              name='level_fusion_downtop_topout')
        down_top_out = fluid.layers.concat(input=[down_out2, top_out], axis=1)
        down_top_out = self.conv_bn_layer(input=down_top_out, num_filters=out_channel, filter_size=1,
                                     stride=1, act='relu',name='level_fusion_downtop')
        return [top_down_out, down_top_out]

    def pyramid_fusion(self,input):
        fusion = fluid.layers.concat(input=input, axis=1)
        out = self.conv_bn_layer(input=fusion, num_filters=2048, filter_size=1,stride=1,
                                 name='pyramid_fusion',act='relu')
        return out

    def AuxHead(self,input, label):

        classdim = self.config['aux_head']['planes']
        weight = self.config['aux_head']['loss_weight']
        drop_ratio = self.config['aux_head']['drop_ratio']
        inplanes = input.shape[1]
        input = self.conv_bn_layer(input=input, num_filters=inplanes * 2, filter_size=[1, 3, 3],
                              stride=[1, 2, 2],padding=[0,1,1],name='AuxHead', act='relu')
        output = fluid.layers.pool3d(input=input, pool_type='avg', global_pooling=True)
        output = fluid.layers.reshape(x=output, shape=[output.shape[0], output.shape[1]])
        output = fluid.layers.dropout(output, drop_ratio, is_test=(not self.is_training),
                                      dropout_implementation='upscale_in_train')
        #将seg_num个输出进行平均融合
        output = fluid.layers.reshape(x=output,shape=[-1, self.seg_num,output.shape[-1]])
        output = fluid.layers.reduce_mean(input=output,dim=1)
        
        fc_in_channel = output.shape[1]
        stdv = 1.0 / math.sqrt(fc_in_channel * 1.0)
        output = fluid.layers.fc(output, size=classdim,param_attr=ParamAttr(name='AuxHead_fc_w',
                                initializer=fluid.initializer.Uniform(-stdv, stdv)),bias_attr=ParamAttr(name='AuxHead_fc_b'))
        loss = fluid.layers.softmax_with_cross_entropy(output, label)
        loss = fluid.layers.reduce_mean(loss,dim=0)
        return loss * weight


