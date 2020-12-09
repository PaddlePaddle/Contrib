
import paddle.fluid as fluid
from paddle.fluid.layers import adaptive_pool2d,reshape,unsqueeze,concat,create_parameter,sqrt,expand,reduce_sum,reduce_mean,transpose,instance_norm
from utils import ReflectionPad2D,ReLU,LeakyReLU,Tanh,Upsample,var,spectral_norm
from paddle.fluid.dygraph import Linear,Conv2D,Layer,Sequential
import math

class ResnetGenerator(Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        self.DownBlock1_1 = ReflectionPad2D(3)
        self.DownBlock1_2 = Conv2D(3, 64, filter_size=7, stride=1, padding=0, bias_attr=False)
        self.DownBlock1_4 = ReLU(False)

        self.DownBlock2_1 = ReflectionPad2D(1)
        self.DownBlock2_2 = Conv2D(64, 128, filter_size=3, stride=2, padding=0, bias_attr=False)
        self.DownBlock2_4 = ReLU(False)

        self.DownBlock3_1 = ReflectionPad2D(1)
        self.DownBlock3_2 = Conv2D(128, 256, filter_size=3, stride=2, padding=0, bias_attr=False)
        self.DownBlock3_4 = ReLU(False)
        n_downsampling = 2
        # Down-Sampling
        self.DownBlock1 = ResnetBlock(256, use_bias=False)
        self.DownBlock2 = ResnetBlock(256, use_bias=False)
        self.DownBlock3 = ResnetBlock(256, use_bias=False)
        self.DownBlock4 = ResnetBlock(256, use_bias=False)
        # Down-Sampling Bottleneck
        mult =4
        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)

        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1,
                                 bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                                     low=-1 / math.sqrt(ngf * mult * 2), high=1 / math.sqrt(ngf * mult * 2))))
        self.relu = ReLU(False)

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(False),
                  Linear(ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(False)]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(False),
                  Linear(ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(False)]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [Upsample(scales=2, resamples='NEAREST'),
                         ReflectionPad2D(1),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         ReLU(False)]

        UpBlock2 += [ReflectionPad2D(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False),
                     Tanh()]

        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):

        x = self.DownBlock1_1(input)
        x = self.DownBlock1_2(x)
        x = instance_norm(x)
        x = self.DownBlock1_4(x)

        x = self.DownBlock2_1(x)
        x = self.DownBlock2_2(x)
        x = instance_norm(x)
        x = self.DownBlock2_4(x)

        x = self.DownBlock3_1(x)
        x = self.DownBlock3_2(x)
        x = instance_norm(x)
        x = self.DownBlock3_4(x)

        gap = adaptive_pool2d(x, 1, pool_type='avg')
        gap_logit = self.gap_fc(reshape(gap, [x.shape[0], -1]))
        gap_weight = self.gap_fc.parameters()[0]
        gap_weight = reshape(gap_weight, shape=[1, -1])
        gap = x * unsqueeze(unsqueeze(gap_weight, 2), 3)

        gmp = adaptive_pool2d(x, 1, pool_type='max')
        gmp_logit = self.gmp_fc(reshape(gmp, [x.shape[0], -1]))
        gmp_weight = self.gmp_fc.parameters()[0]
        gmp_weight = reshape(gmp_weight, shape=[1, -1])
        gmp = x * unsqueeze(unsqueeze(gmp_weight, 2), 3)

        cam_logit = concat([gap_logit, gmp_logit], 1)
        x = concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = adaptive_pool2d(x, 1, pool_type='avg')
            x_ = self.FC(reshape(x_, [x_.shape[0], -1]))
        else:
            x_ = self.FC(reshape(x, [x.shape[0], -1]))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()

        self.conv_block1_1 = ReflectionPad2D(1)
        self.conv_block1_2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.conv_block1_4 = ReLU(False)

        self.conv_block2_1 = ReflectionPad2D(1)
        self.conv_block2_2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)


    def forward(self, x):
        y = self.conv_block1_1(x)
        y = self.conv_block1_2(y)
        y = instance_norm(y)
        y = self.conv_block1_4(y)
        y = self.conv_block2_1(y)
        y = self.conv_block2_2(y)
        y = instance_norm(y)
        out = x + y
        return out


class ResnetAdaILNBlock(Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2D(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU(False)

        self.pad2 = ReflectionPad2D(1)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = create_parameter(shape=[1, num_features, 1, 1], dtype='float32',is_bias=True,
                                    default_initializer=fluid.initializer.Constant(0.9))

    def forward(self, input, gamma, beta):
        in_mean, in_var = reduce_mean(input, dim=[2, 3], keep_dim=True), var(input, dim=[2, 3], keep_dim=True)
        out_in = (input - in_mean) / sqrt(in_var + self.eps)
        ln_mean, ln_var = reduce_mean(input, dim=[1, 2, 3], keep_dim=True), var(input, dim=[1, 2, 3], keep_dim=True)
        out_ln = (input - ln_mean) / sqrt(ln_var + self.eps)
        out = expand(self.rho,expand_times=[input.shape[0], 1, 1, 1]) * out_in + (
                1 - expand(self.rho,expand_times=[input.shape[0], 1, 1, 1])) * out_ln
        out = out * unsqueeze(unsqueeze(gamma, 2), 3) + unsqueeze(unsqueeze(beta, 2), 3)

        return out


class ILN(Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = create_parameter(shape=[1, num_features, 1, 1], dtype='float32',is_bias=True,
                                    default_initializer=fluid.initializer.Constant(0.0))
        self.gamma = create_parameter(shape=[1, num_features, 1, 1], dtype='float32',is_bias=True,
                                      default_initializer=fluid.initializer.Constant(1.0))
        self.beta = create_parameter(shape=[1, num_features, 1, 1], dtype='float32',is_bias=True,
                                     default_initializer=fluid.initializer.Constant(0.0))

    def forward(self, input):
        in_mean, in_var = reduce_mean(input, dim=[2, 3], keep_dim=True), var(input, dim=[2, 3], keep_dim=True)
        out_in = (input - in_mean) / sqrt(in_var + self.eps)
        ln_mean, ln_var = reduce_mean(input, dim=[1, 2, 3], keep_dim=True), var(input, dim=[1, 2, 3], keep_dim=True)
        out_ln = (input - ln_mean) / sqrt(ln_var + self.eps)
        out = expand(self.rho,expand_times=[input.shape[0], 1, 1, 1]) * out_in + (
                1 - expand(self.rho,expand_times=[input.shape[0], 1, 1, 1])) * out_ln
        out = out * expand(self.gamma,expand_times=[input.shape[0], 1, 1, 1]) + expand(self.beta,expand_times=[input.shape[0], 1, 1, 1])

        return out


class Discriminator(Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2D(1),
                 spectral_norm(
                     Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0,
                            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                                low=-1 / math.sqrt(input_nc * 16), high=1 / math.sqrt(input_nc * 16))))),
                 LeakyReLU(0.2, False)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2D(1),
                      spectral_norm(
                          Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0,
                                 bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                                     low=-1 / math.sqrt(ndf * mult * 16), high=1 / math.sqrt(ndf * mult * 16))))),
                      LeakyReLU(0.2, False)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2D(1),
                  spectral_norm(
                      Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0,
                             bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                                 low=-1 / math.sqrt(ndf * mult * 16), high=1 / math.sqrt(ndf * mult * 16))))),
                  LeakyReLU(0.2, False)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = spectral_norm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = spectral_norm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1,
                              bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                                  low=-1 / math.sqrt(ndf * mult * 2), high=1 / math.sqrt(ndf * mult * 2))))
        self.leaky_relu = LeakyReLU(0.2, False)

        self.pad = ReflectionPad2D(1)
        self.conv = spectral_norm(
            Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False))

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = adaptive_pool2d(x, 1, pool_type='avg')
        gap_logit = self.gap_fc(reshape(gap, shape=[x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = transpose(gap_weight, perm=[1,0])
        gap = x * unsqueeze(unsqueeze(gap_weight, 2), 3)

        gmp = adaptive_pool2d(x, 1, pool_type='max')
        gmp_logit = self.gmp_fc(reshape(gmp, shape=[x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = transpose(gmp_weight, perm=[1,0])
        gmp = x * unsqueeze(unsqueeze(gmp_weight, 2), 3)

        cam_logit = concat([gap_logit, gmp_logit], 1)
        x = concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap
