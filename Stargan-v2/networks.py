from ops import *
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Sequential
from paddle.fluid.dygraph.nn import Conv2D, InstanceNorm


class Generator(fluid.dygraph.Layer):
    def __init__(self, img_size=256, img_ch=3, style_dim=64, max_conv_dim=512, sn=False, w_hpf=0):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.style_dim = style_dim
        self.max_conv_dim = max_conv_dim
        self.sn = sn
        self.channels = 2 ** 14 // img_size  # if 256 -> 64
        self.w_hpf = w_hpf
        self.repeat_num = int(np.log2(img_size)) - 4  # if 256 -> 4
        if self.w_hpf == 1:
            self.repeat_num += 1
        self.from_rgb = Conv2D(num_channels=self.img_ch, num_filters=self.channels, filter_size=3,
                               padding=1, param_attr=weight_initializer, bias_attr=bias_initializer,
                               stride=1, act=None)

        self.encode, self.decode, self.to_rgb = self.architecture_init()

    def architecture_init(self):
        ch_in = self.channels
        ch_out = self.channels
        encoder = []
        decoder = []
        # down/up-sampling blocks
        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)
            encoder.append(ResBlock(ch_in, ch_out, normalize=True, downsample=True, sn=self.sn))
            decoder.insert(0, AdainResBlock(ch_out, ch_in, upsample=True, sn=self.sn, w_hpf=self.w_hpf))  # stack-like
            ch_in = ch_out
        # bottleneck blocks
        for i in range(2):
            encoder.append(ResBlock(ch_out, ch_out, normalize=True, sn=self.sn))
            decoder.insert(0, AdainResBlock(ch_out, ch_out, sn=self.sn, w_hpf=self.w_hpf))
        to_rgb_layer = []
        to_rgb_layer.append(InstanceNorm(self.style_dim))
        to_rgb_layer.append(Leaky_Relu(alpha=0.2))
        to_rgb_layer.append(Conv2D(num_channels=self.style_dim, num_filters=self.img_ch, filter_size=1,
                                   padding=0, param_attr=weight_initializer, bias_attr=bias_initializer,
                                   stride=1, act=None))

        encoders = Sequential(*encoder)
        decoders = Sequential(*decoder)
        to_rgb = Sequential(*to_rgb_layer)

        return encoders, decoders, to_rgb

    def forward(self, x_init, training=True, masks=None):
        if self.w_hpf > 0:
            self.hpf = HighPass(self.w_hpf)
        x, x_s = x_init
        x = self.from_rgb(x)
        cache = {}
        for i in range(len(self.encode)):
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = self.encode[i](x)
        # x = self.encode(x)
        for i in range(len(self.decode)):
            x = self.decode[i]([x, x_s])
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = fluid.layers.image_resize(mask, scale=x.shape[2] / mask.shape[2], resample='BILINEAR')
                tmp = fluid.layers.elementwise_mul(mask, cache[x.shape[2]])
                mask = self.hpf(tmp)
                x = x + mask  # self.hpf(mask * cache[x.shape[2]])

        x = self.to_rgb(x)
        # print('Generator final x:', x.shape,)
        return x


class MappingNetwork(fluid.dygraph.Layer):
    def __init__(self, style_dim=64, hidden_dim=512, num_domains=2, sn=False):
        super(MappingNetwork, self).__init__()
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.sn = sn
        self.latent_dim = 16
        self.shared, self.unshared = self.architecture_init()

    def architecture_init(self):
        shared_sub_layers = [Linear(self.latent_dim, self.hidden_dim, param_attr=weight_initializer,
                                    bias_attr=bias_initializer, act=None)]
        shared_sub_layers.append(Relu())
        for i in range(3):
            shared_sub_layers.append(Linear(self.hidden_dim, self.hidden_dim, param_attr=weight_initializer,
                                            bias_attr=bias_initializer, act=None))
            shared_sub_layers.append(Relu())
        shared_layers = Sequential(*shared_sub_layers)
        unshared_layer = []
        for n_d in range(self.num_domains):
            unshared_sub_layers = []
            for i in range(3):
                unshared_sub_layers.append(Linear(self.hidden_dim, self.hidden_dim, param_attr=weight_initializer,
                                                  bias_attr=bias_initializer, act=None))
                unshared_sub_layers.append(Relu())
            unshared_sub_layers.append(Linear(self.hidden_dim, self.style_dim, param_attr=weight_initializer,
                                              bias_attr=bias_initializer, act=None))
            unshared_layer.append(Sequential(*unshared_sub_layers))
        unshared_layers = Sequential(*unshared_layer)

        return shared_layers, unshared_layers

    def forward(self, x_init, training=True, mask=None):
        z, domain = x_init
        # print('MappingNetwork z domain:',z.shape,domain.shape)
        h = self.shared(z)
        x = []
        for i in range(self.num_domains):
            x += [self.unshared[i](h)]

        # 这里可以改成把x转成(batch,len(domain))的tensor,然后输出(batch,domain)  索引domain位置输出(4,64)
        x = paddle.fluid.layers.stack(x, axis=1)  # [bs, num_domains, style_dim]
        batch_size = int(x.shape[0])
        o = []
        for i in range(batch_size):
            data_index = np.array([i]).astype('int32')
            index = fluid.dygraph.to_variable(data_index)
            dex = fluid.dygraph.to_variable(domain[i])
            w = paddle.fluid.layers.index_select(x, index, dim=0)
            o += [fluid.layers.reshape(paddle.fluid.layers.index_select(w, dex, dim=1), shape=[64])]
        o = paddle.fluid.layers.stack(o, axis=0)
        # print("MappingNetwork final x",o.shape)
        # print(o.numpy())
        return o  # (4, 64)


class StyleEncoder(fluid.dygraph.Layer):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512, sn=False):
        super(StyleEncoder, self).__init__()
        self.img_size = img_size
        self.style_dim = style_dim
        self.num_domains = num_domains
        self.max_conv_dim = max_conv_dim
        self.sn = sn
        self.channels = 2 ** 14 // img_size  # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 2  # if 256 -> 6
        self.shared, self.unshared = self.architecture_init()

    def architecture_init(self):
        # shared layers
        ch_in = self.channels
        shared_sub_layers = [Conv2D(num_channels=3, num_filters=ch_in, filter_size=3,
                                    padding=1, param_attr=weight_initializer, bias_attr=bias_initializer,
                                    stride=1, act=None)]
        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)
            sub_layer = ResBlock(ch_in, ch_out, normalize=False, downsample=True, sn=self.sn, act=None)
            ch_in = ch_out
            shared_sub_layers.append(sub_layer)
        shared_sub_layers.append(Leaky_Relu(alpha=0.2))
        shared_sub_layers.append(Conv2D(num_channels=self.max_conv_dim, num_filters=self.max_conv_dim,
                                        filter_size=4, padding=0, param_attr=weight_initializer,
                                        bias_attr=bias_initializer,
                                        stride=1, act=None))
        shared_sub_layers.append(Leaky_Relu(alpha=0.2))

        shared_layers = Sequential(*shared_sub_layers)
        # unshared layers
        unshared_sub_layers = []
        for _ in range(self.num_domains):
            sub_layer = Linear(self.max_conv_dim, self.style_dim, param_attr=weight_initializer,
                               bias_attr=bias_initializer, act=None)
            unshared_sub_layers.append(sub_layer)

        unshared_layers = Sequential(*unshared_sub_layers)
        return shared_layers, unshared_layers

    def forward(self, x_init, training=True, mask=None):
        x, domain = x_init
        x = self.shared(x)
        h = x

        h = fluid.layers.reshape(h, shape=[-1, self.max_conv_dim])

        z = []
        for i in range(self.num_domains):
            z += [self.unshared[i](h)]
        z = paddle.fluid.layers.stack(z, axis=1)  # [bs, num_domains, style_dim]
        #print('z',z.numpy())
        batch_size = int(x.shape[0])
        o = []
        for i in range(batch_size):
            data_index = np.array([i]).astype('int32')
            index = fluid.dygraph.to_variable(data_index)
            dex = fluid.dygraph.to_variable(domain[i])
            w = paddle.fluid.layers.index_select(z, index, dim=0)
            o += [fluid.layers.reshape(paddle.fluid.layers.index_select(w, dex, dim=1), shape=[64])]

        o = paddle.fluid.layers.stack(o, axis=0)

        # print("StyleEncoder final x", o.shape)  #[batch,64]
        #print(o.numpy())
        return o


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, sn=False):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.num_domains = num_domains
        self.max_conv_dim = max_conv_dim
        self.sn = sn
        self.channels = 2 ** 14 // img_size  # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 2  # if 256 -> 6
        self.main = self.architecture_init()

    def architecture_init(self):
        ch_in = self.channels
        sub_layers = [Conv2D(num_channels=3, num_filters=self.channels, filter_size=3, padding=1,
                             param_attr=weight_initializer, bias_attr=bias_initializer,
                             stride=1, act=None)]
        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)
            sub_layer = ResBlock(ch_in, ch_out, normalize=False, downsample=True, sn=self.sn, act=None)
            ch_in = ch_out
            sub_layers.append(sub_layer)
        sub_layers.append(Leaky_Relu(alpha=0.2))
        sub_layers.append(Conv2D(num_channels=self.max_conv_dim, num_filters=self.max_conv_dim, filter_size=4,
                                 padding=0, param_attr=weight_initializer, bias_attr=bias_initializer,
                                 stride=1, act=None))
        sub_layers.append(Leaky_Relu(alpha=0.2))
        sub_layers.append(Conv2D(num_channels=self.max_conv_dim, num_filters=self.num_domains, filter_size=1,
                                 padding=0, param_attr=weight_initializer, bias_attr=bias_initializer,
                                 stride=1, act=None))

        encoder = Sequential(*sub_layers)
        return encoder

    def forward(self, x_init, training=True, mask=None):
        x, domain = x_init
        x = self.main(x)
        batch_size = int(x.shape[0])
        o = []
        for i in range(batch_size):
            data_index = np.array([i]).astype('int32')
            index = fluid.dygraph.to_variable(data_index)
            dex = fluid.dygraph.to_variable(domain[i])
            w = paddle.fluid.layers.index_select(x, index, dim=0)
            o += [fluid.layers.reshape(paddle.fluid.layers.index_select(w, dex, dim=1), shape=[1])]

        o = paddle.fluid.layers.stack(o, axis=0)
        # print('Discriminator final',o.shape)  #(batch,domian)
        # print(o.numpy())
        return o

# with fluid.dygraph.guard():
#     import matplotlib.pyplot as plt
#     import cv2
#
#     img = cv2.imread('002140.jpg')
#     img = cv2.resize(img, (256, 256))
#     img_ = np.array(img).astype('float32')
#     img_ = img_.transpose([2, 0, 1])
#     img_ = fluid.dygraph.to_variable(img_)
#     img_ = fluid.layers.unsqueeze(img_, 0)
#     print(img_.shape)
#     fan = FAN(fname_pretrained='fan')
#
#     # print(len(fan.state_dict().keys()))
#     # print(fan.state_dict().keys())
#
#     masks = fan.get_heatmap(img_)
#     print('ssssssssssssssssssssssssssssssss')
#     print(masks)
