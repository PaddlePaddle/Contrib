from paddle.fluid.dygraph import Conv2D, Layer, Pool2D, Linear, Sequential, BatchNorm
from paddle.fluid.layers import concat

import paddle.fluid as fluid
import numpy as np
import copy
from model import Res3D

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y

class LinConPoo(Layer):

    def __init__(self, sequence_list):

        super(LinConPoo, self).__init__()
        self.__sequence_list = copy.deepcopy(sequence_list)

        if not isinstance(self.__sequence_list, list): raise ValueError('sequence_list error')

        self._layers_squence = Sequential()
        self._layers_list = []

        LAYLIST = [ConvBNLayer, Conv2D, Linear, Pool2D]
        for i, layer_arg in enumerate(self.__sequence_list):

            if isinstance(layer_arg, dict):

                layer_class = layer_arg.pop('type')

                if not layer_class in LAYLIST:
  
                    raise KeyError("the parameters of sequence_list must be within `[ConvBNLayer, Conv2D, Linear, Pool2D]`")

                layer_obj = layer_class(**layer_arg)


            elif isinstance(layer_arg, list):

                layer_class = layer_arg.pop(0)

                if not layer_class in LAYLIST:

                    raise KeyError("the parameters of sequence_list must be within `[ConvBNLayer, Conv2D, Linear, Pool2D]`")

                layer_obj = layer_class(*layer_arg)


            else:
                raise ValueError("sequence_list error")


            layer_name = layer_class.__name__ + str(i)

            self._layers_list.append((layer_name, layer_obj))

            self._layers_squence.add_sublayer(*(layer_name, layer_obj))

        self._layers_squence = Sequential(*self._layers_list)


    def forward(self, inputs, show_shape=False):

        if show_shape:

            x = inputs
            for op in self._layers_list:
                x = op[1](x)
                print(op[0], '\t', x.shape)
            return x

        return self._layers_squence(inputs)


class Inception(fluid.dygraph.Layer):
    
    def __init__(self, num_channels, ch1x1, ch3x3reduced, ch3x3, doublech3x3reduced, doublech3x3_1, doublech3x3_2, pool_proj):
        '''
        @Brief
             `Inception` 
 
        @Parameters
            num_channels : channel numbers of input tensor
            ch1x1        : output channel numbers of 1x1 conv
            ch3x3reduced : channel numbers of 1x1 conv before 3x3 conv
            ch3x3        : output channel numbers of 3x3 conv
            doublech3x3reduced : channel numbers of 1x1 conv before the double 3x3 convs
            doublech3x3_1        : output channel numbers of the first 3x3 conv
            doublech3x3_2        : output channel numbers of the second 3x3 conv
            pool_proj    : output channel numbers of 1x1 conv after pool

        @Return
             `Inception` model

        '''

        super(Inception, self).__init__()

        branch1_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':ch1x1, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
        ]
        self.branch1 = LinConPoo(branch1_list)
        
        branch2_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':ch3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':ch3x3reduced,  'num_filters':ch3x3,        'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
        ]
        self.branch2 = LinConPoo(branch2_list)
        
        branch3_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':doublech3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':doublech3x3reduced,  'num_filters':doublech3x3_1,  'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':doublech3x3_1,  'num_filters':doublech3x3_2,  'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
        ]
        self.branch3 = LinConPoo(branch3_list)
        
        branch4_list = [
                {'type':Pool2D,  'pool_size':3,  'pool_type':'avg',  'pool_stride':1,  'pool_padding':1,  'global_pooling':False},
                {'type':ConvBNLayer,  'num_channels':num_channels, 'num_filters':pool_proj, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
        ]
        self.branch4 = LinConPoo(branch4_list)

    
    
    def forward(self, inputs):
        '''
        @Parameters :
            inputs: input tensor
        '''

        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)

        outputs = concat([branch1, branch2, branch3, branch4], axis=1)

        return outputs

class Inception3c(fluid.dygraph.Layer):
    
    def __init__(self, num_channels, ch3x3reduced, ch3x3, doublech3x3reduced, doublech3x3_1, doublech3x3_2):
        '''
        @Brief
             `Inception3c`
 
        @Parameters
            num_channels : channel numbers of input tensor
            ch3x3reduced : channel numbers of 1x1 conv before 3x3 conv
            ch3x3        : output channel numbers of 3x3 conv
            doublech3x3reduced : channel numbers of 1x1 conv before the double 3x3 convs
            doublech3x3_1        : output channel numbers of the first 3x3 conv
            doublech3x3_2        : output channel numbers of the second 3x3 conv

        @Return
            `Inception_3c` model

        '''

        super(Inception3c, self).__init__()

        
        branch1_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':ch3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':ch3x3reduced,  'num_filters':ch3x3,        'filter_size':3, 'stride':2, 'padding':1, 'act':'relu'},
        ]
        self.branch1 = LinConPoo(branch1_list)
        
        branch2_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':doublech3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':doublech3x3reduced,  'num_filters':doublech3x3_1,  'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
                
        ]
        self.branch2 = LinConPoo(branch2_list)

        branch3_list = [
            {'type':ConvBNLayer, 'num_channels':doublech3x3_1,  'num_filters':doublech3x3_2,  'filter_size':3, 'stride':2, 'padding':1, 'act':'relu'},
        ]

        self.branch3 = LinConPoo(branch3_list)
        
        branch4_list = [
                {'type':Pool2D,  'pool_size':3,  'pool_type':'max',  'pool_stride':2,  'pool_padding':1,  'global_pooling':False},
        ]
        self.branch4 = LinConPoo(branch4_list)
    
    def forward(self, inputs):
        '''
        @Parameters :
            inputs: input tensor
        '''

        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(branch2)
        branch4 = self.branch4(inputs)
        
        outputs = concat([branch1, branch3, branch4], axis=1)

        return outputs, branch2


class Inception4e(fluid.dygraph.Layer):
    
    def __init__(self, num_channels, ch3x3reduced, ch3x3, doublech3x3reduced, doublech3x3_1, doublech3x3_2, pool_proj):
        '''
        @Brief
            `Inception4e`
 
        @Parameters
            num_channels : channel numbers of input tensor
            ch1x1        : output channel numbers of 1x1 conv
            ch3x3reduced : channel numbers of 1x1 conv before 3x3 conv
            ch3x3        : output channel numbers of 3x3 conv
            doublech3x3reduced : channel numbers of 1x1 conv before the double 3x3 convs
            doublech3x3_1        : output channel numbers of the first 3x3 conv
            doublech3x3_2        : output channel numbers of the second 3x3 conv
            pool_proj    : output channel numbers of 1x1 conv after pool

        @Return
            `Inception_4e`

        '''

        super(Inception4e, self).__init__()
        
        branch1_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':ch3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':ch3x3reduced,  'num_filters':ch3x3,        'filter_size':3, 'stride':2, 'padding':1, 'act':'relu'},
        ]
        self.branch1 = LinConPoo(branch1_list)
        
        branch2_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':doublech3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':doublech3x3reduced,  'num_filters':doublech3x3_1,  'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':doublech3x3_1,  'num_filters':doublech3x3_2,  'filter_size':3, 'stride':2, 'padding':1, 'act':'relu'},
        ]
        self.branch2 = LinConPoo(branch2_list)
        
        branch3_list = [
                {'type':Pool2D,  'pool_size':3,  'pool_type':'max',  'pool_stride':2,  'pool_padding':1,  'global_pooling':False},
        ]
        self.branch3 = LinConPoo(branch3_list)

    def forward(self, inputs):
        '''
        @Parameters :
            inputs: input tensor
        '''

        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        
        outputs = concat([branch1, branch2, branch3], axis=1)

        return outputs


class Inception5a(fluid.dygraph.Layer):
    
    def __init__(self, num_channels, ch1x1, ch3x3reduced, ch3x3, doublech3x3reduced, doublech3x3_1, doublech3x3_2, pool_proj):
        '''
        @Brief
            `Inception5a`
 
        @Parameters
            num_channels : channel numbers of input tensor
            ch1x1        : output channel numbers of 1x1 conv
            ch3x3reduced : channel numbers of 1x1 conv before 3x3 conv
            ch3x3        : output channel numbers of 3x3 conv
            doublech3x3reduced : channel numbers of 1x1 conv before the double 3x3 convs
            doublech3x3_1        : output channel numbers of the first 3x3 conv
            doublech3x3_2        : output channel numbers of the second 3x3 conv
            pool_proj    : output channel numbers of 1x1 conv after pool

        @Return
            `Inception_5a` model

        '''

        super(Inception5a, self).__init__()

        branch1_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':ch1x1, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
        ]

        self.branch1 = LinConPoo(branch1_list)

        branch2_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':ch3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':ch3x3reduced,  'num_filters':ch3x3,        'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
        ]
        self.branch2 = LinConPoo(branch2_list)
        
        branch3_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':doublech3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':doublech3x3reduced,  'num_filters':doublech3x3_1,  'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':doublech3x3_1,  'num_filters':doublech3x3_2,  'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
        ]
        self.branch3 = LinConPoo(branch3_list)
        
        branch4_list = [
                {'type':Pool2D,  'pool_size':3,  'pool_type':'avg',  'pool_stride':1,  'pool_padding':1,  'global_pooling':False},
                {'type':ConvBNLayer,  'num_channels':num_channels, 'num_filters':pool_proj, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
        ]
        self.branch4 = LinConPoo(branch4_list)

    def forward(self, inputs):
        '''
        @Parameters :
            inputs: input tensor
        '''
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        
        outputs = concat([branch1, branch2, branch3, branch4], axis=1)

        return outputs

class Inception5b(fluid.dygraph.Layer):
    
    def __init__(self, num_channels, ch1x1, ch3x3reduced, ch3x3, doublech3x3reduced, doublech3x3_1, doublech3x3_2, pool_proj):
        '''
        @Brief
            `Inception5b`
 
        @Parameters
            num_channels : channel numbers of input tensor
            ch1x1        : output channel numbers of 1x1 conv
            ch3x3reduced : channel numbers of 1x1 conv before 3x3 conv
            ch3x3        : output channel numbers of 3x3 conv
            doublech3x3reduced : channel numbers of 1x1 conv before the double 3x3 convs
            doublech3x3_1        : output channel numbers of the first 3x3 conv
            doublech3x3_2        : output channel numbers of the second 3x3 conv
            pool_proj    : output channel numbers of 1x1 conv after pool

        @Return
            `Inception_5b` model

        '''

        super(Inception5b, self).__init__()

        branch1_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':ch1x1, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
        ]

        self.branch1 = LinConPoo(branch1_list)

        branch2_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':ch3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':ch3x3reduced,  'num_filters':ch3x3,        'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
        ]
        self.branch2 = LinConPoo(branch2_list)
        
        branch3_list = [
                {'type':ConvBNLayer, 'num_channels': num_channels, 'num_filters':doublech3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':doublech3x3reduced,  'num_filters':doublech3x3_1,  'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
                {'type':ConvBNLayer, 'num_channels':doublech3x3_1,  'num_filters':doublech3x3_2,  'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
        ]
        self.branch3 = LinConPoo(branch3_list)
        
        branch4_list = [
                {'type':Pool2D,  'pool_size':3,  'pool_type':'max',  'pool_stride':1,  'pool_padding':1,  'global_pooling':False},
                {'type':ConvBNLayer,  'num_channels':num_channels, 'num_filters':pool_proj, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
        ]
        self.branch4 = LinConPoo(branch4_list)

    def forward(self, inputs):
        '''
        @Parameters :
            inputs: input tensor
        '''

        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        
        outputs = concat([branch1, branch2, branch3, branch4], axis=1)

        return outputs

class GoogLeNet(fluid.dygraph.Layer):

    def __init__(self, class_dim=101, seg_num=24, seglen=1, modality="RGB"):
        '''
        @Brief:
            `GoogLeNet` model
            input image should be 224 * 224
        @Parameters:
            num_channels : channel numbers of input tensor
            out_dim      : the number of classes for classification
        @Return:
            out          : shape=(X, class_dim)

        >>> import numpy as np
        >>> data = np.ones(shape=(8, 3, 224, 224), dtype=np.float32)
        >>> with fluid.dygraph.guard():
                googlenet = GoogLeNet(class_dim=10)
                data = fluid.dygraph.to_variable(data)
                y = googlenet(data)
                print(y.numpy().shape)
        (8, 10)
        ''' 
        self.seg_num = seg_num
        self.seglen = seglen
        self.modality = modality
        self.channels = 3 * self.seglen if self.modality == "RGB" else 2 * self.seglen

        super(GoogLeNet, self).__init__() 
        
        part1_list  = [
            {'type':ConvBNLayer, 'num_channels':self.channels, 'num_filters':64, 'filter_size':7, 'stride':2, 'padding':3, 'act':'relu'},
            {'type':Pool2D, 'pool_size':3, 'pool_type':'max', 'pool_stride':2, 'pool_padding':1, 'global_pooling':False},
        ]
        
        part2_list  = [
            {'type':ConvBNLayer, 'num_channels':64, 'num_filters':64 , 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu'},
            {'type':ConvBNLayer, 'num_channels':64, 'num_filters':192, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu'},
            {'type':Pool2D, 'pool_size':3, 'pool_type':'max', 'pool_stride':2, 'pool_padding':1, 'global_pooling':False},
        ]

        ##the values of the two pool_padding layers above  are changed from 0 to 1 in order to comply with 28x28 in the paper。However it is 27x27 in the original Caffe code
    
        self.googLeNet_part1 = Sequential(
                                ('part1', LinConPoo(part1_list)), 
                                ('part2', LinConPoo(part2_list)), 
                                ('inception_3a', Inception(192,  64,  64, 64, 64, 96, 96, 32)), 
                                ('inception_3b', Inception(256, 64, 64, 96, 64, 96, 96, 64)), 
                            )

        self.before3d = Sequential(
            ('Inception3c', Inception3c(320, 128, 160, 64, 96, 96)) 
        ) 

        self.googLeNet_part2 = Sequential(
                                ('inception_4a', Inception(576, 224, 64, 96, 96, 128, 128, 128)),
                                ('inception_4b', Inception(576, 192, 96, 128, 96, 128, 128, 128)),
                                ('inception_4c', Inception(576, 160, 128, 160, 128, 160, 160, 128)), 
                                ('inception_4d', Inception(608, 96, 128, 192, 160, 192, 192, 128)), 
                            )

        self.googLeNet_part3 = Sequential(
                                ('inception_4e', Inception4e(608, 128, 192, 192, 256, 256, 608)), 
                                ('inception_5a', Inception5a(1056, 352, 192, 320, 160, 224, 224, 128)), 
                                ('inception_5b', Inception5b(1024, 352, 192, 320, 192, 224, 224, 128)), 
                                ('AvgPool1', Pool2D(pool_size=7, pool_type='avg', pool_stride=1, global_pooling=True)), 
                            )

        self.res3d = Res3D.ResNet3D('resnet', channels=96) 

        self.out = Linear(input_dim=1536,
                          output_dim=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Xavier(uniform=False)))


        self.out_3d = []

    def forward(self, inputs, label=None):

        inputs = fluid.layers.reshape(inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])

        googLeNet_part1 = self.googLeNet_part1(inputs)

        googleNet_b3d, before3d = self.before3d(googLeNet_part1)

        if len(self.out_3d) == self.seg_num:
            
            self.out_3d[:self.seg_num - 1] = self.out_3d[1:]
            self.out_3d[self.seg_num - 1] = before3d
            for input_old in self.out_3d[:self.seg_num - 1]:
                input_old.stop_gradient = True
        else:
            while len(self.out_3d) < self.seg_num:
                self.out_3d.append(before3d)

        y_out_3d = self.out_3d[0]
        for i in range(len(self.out_3d) - 1):
            y_out_3d = fluid.layers.concat(input=[y_out_3d,self.out_3d[i+1]], axis=0)

        y_out_3d = fluid.layers.reshape(y_out_3d, [-1, self.seg_num, y_out_3d.shape[1], y_out_3d.shape[2], y_out_3d.shape[3]])
 
        y_out_3d = fluid.layers.reshape(y_out_3d, [y_out_3d.shape[0], y_out_3d.shape[2], y_out_3d.shape[1], y_out_3d.shape[3], y_out_3d.shape[4]])
        
        out_final_3d = self.res3d(y_out_3d)

        out_final_3d = fluid.layers.reshape(out_final_3d, [-1, out_final_3d.shape[1]])

        out_final_3d = fluid.layers.dropout(out_final_3d, dropout_prob=0.5)

        out_final_3d = fluid.layers.reshape(out_final_3d, [-1, self.seg_num, out_final_3d.shape[1]])

        out_final_3d = fluid.layers.reduce_mean(out_final_3d, dim=1)

        googLeNet_part2 = self.googLeNet_part2(googleNet_b3d)

        googLeNet_part3 = self.googLeNet_part3(googLeNet_part2)

        googLeNet_part3 = fluid.layers.dropout(googLeNet_part3, dropout_prob=0.6)

        out_final_2d = fluid.layers.reshape(googLeNet_part3, [-1, googLeNet_part3.shape[1]])

        out_final_2d = fluid.layers.reshape(out_final_2d, [-1, self.seg_num, out_final_2d.shape[1]])

        out_final_2d = fluid.layers.reduce_mean(out_final_2d, dim=1)

        out_final = fluid.layers.concat(input=[out_final_2d,out_final_3d], axis=1)

        out_final = self.out(out_final)
        
        if label is not None:
            acc = fluid.layers.accuracy(input=out_final, label=label)
            return out_final, acc
        else:
            return out_final


if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = GoogLeNet()
        img = np.zeros([1, 2, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img)
        print(outs.shape)

