import paddle
import paddle.fluid as fluid
import math
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm, InstanceNorm
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph import Sequential
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.base import to_variable
from copy import deepcopy
from functools import partial
from munch import Munch
from collections import namedtuple

##################################################################################
# Initialization
##################################################################################

# factor, mode = pytorch_kaiming_weight_factor(activation_function='relu')
# distribution = "untruncated_normal"
# distribution in {"uniform", "truncated_normal", "untruncated_normal"}
# weight_initializer = fluid.initializer.Normal(loc=0.0, scale=0.05)  # scale=factor

# weight_initializer =fluid.initializer.Normal(0., 0.05)
weight_initializer = fluid.initializer.MSRAInitializer(uniform=False)
# weight_initializer = fluid.initializer.Xavier(uniform=False)
bias_initializer = ParamAttr(initializer=fluid.initializer.Constant(0.0))
bias_initializer_1x1 = False
AdaIN_initializer = False
weight_regularizer = None  # fluid.regularizer.L2Decay(regularization_coeff=1e-4)  # 0.1
weight_regularizer_fully = None  # fluid.regularizer.L2Decay(regularization_coeff=1e-4)  # 0.1


##################################################################################
# Layers
##################################################################################

class HourGlass(fluid.dygraph.Layer):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(64, 64, True, True, 256, first_one,
                                     out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_sublayer('b1_' + str(level), ConvBlock(256, 256))
        self.add_sublayer('b2_' + str(level), ConvBlock(256, 256))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_sublayer('b2_plus_' + str(level), ConvBlock(256, 256))
        self.add_sublayer('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._sub_layers['b1_' + str(level)](up1)
        low1 = fluid.layers.pool2d(inp, pool_size=2, pool_type="avg", pool_stride=2)
        low1 = self._sub_layers['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._sub_layers['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._sub_layers['b3_' + str(level)](low3)
        up2 = fluid.layers.image_resize(low3, scale=2, resample='NEAREST')

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class AddCoordsTh(fluid.dygraph.Layer):
    def __init__(self, height=64, width=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary

        with fluid.dygraph.no_grad():
            x_coords = fluid.layers.arange(0, height)
            x_coords = fluid.layers.unsqueeze(x_coords, 1)
            x_coords = fluid.layers.expand(x_coords, [1, width])

            y_coords = fluid.layers.arange(0, width)
            y_coords = fluid.layers.unsqueeze(y_coords, 0)
            y_coords = fluid.layers.expand(y_coords, [height, 1])

            x_coords = (x_coords / (height - 1)) * 2 - 1
            y_coords = (y_coords / (width - 1)) * 2 - 1
            coords = fluid.layers.stack([x_coords, y_coords], axis=0)  # (2, height, width)
            if self.with_r:
                rr = fluid.layers.sqrt(fluid.layers.pow(x_coords, 2) + fluid.layers.pow(y_coords, 2))  # (height, width)
                rr = fluid.layers.unsqueeze((rr / fluid.layers.reduce_max(rr)), 0)
                coords = fluid.layers.concat([coords, rr], axis=0)

            self.coords = fluid.layers.unsqueeze(coords, axes=0)  # (1, 2 or 3, height, width)

            self.x_coords = x_coords
            self.y_coords = y_coords

    def forward(self, x, heatmap=None):
        """
        x: (batch, c, x_dim, y_dim)
        """
        coords = self.coords.numpy().repeat(x.shape[0], 1)
        coords = fluid.dygraph.to_variable(coords)

        if self.with_boundary and heatmap is not None:
            heatmap = heatmap.numpy()
            boundary_channel = np.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)

            print('boundary_channel', boundary_channel)
            zero_tensor = fluid.layers.zeros_like(self.x_coords)

            zero_tensor = zero_tensor.numpy()
            self.x_coords = self.x_coords.numpy()
            self.y_coords = self.y_coords.numpy()
            print(self.x_coords)
            print(self.y_coords)
            xx_boundary_channel = np.where(boundary_channel > 0.05, self.x_coords, zero_tensor)
            yy_boundary_channel = np.where(boundary_channel > 0.05, self.y_coords, zero_tensor)
            xx_boundary_channel = fluid.dygraph.to_variable(xx_boundary_channel)
            yy_boundary_channel = fluid.dygraph.to_variable(yy_boundary_channel)

            coords = fluid.layers.concat([coords, xx_boundary_channel, yy_boundary_channel], axis=1)

        coords = fluid.layers.reshape(coords, [x.shape[0], 3, coords.shape[2], coords.shape[3]])

        x_and_coords = fluid.layers.concat([x, coords], axis=1)
        return x_and_coords


class CoordConvTh(fluid.dygraph.Layer):
    """CoordConv layer as in the paper."""

    def __init__(self, height, width, with_r, with_boundary,
                 in_channels, first_one=False, out_channels=256,
                 kernel_size=1, stride=1, padding=0):
        super(CoordConvTh, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.addcoords = AddCoordsTh(height, width, with_r, with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = Conv2D(num_channels=in_channels, num_filters=self.out_channels,
                           filter_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


class ConvBlock(fluid.dygraph.Layer):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = BatchNorm(in_planes)
        conv3x3 = partial(Conv2D, filter_size=3, stride=1, padding=1, bias_attr=bias_initializer_1x1)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = BatchNorm(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = BatchNorm(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.downsample = None
        if in_planes != out_planes:
            self.downsample = self.architecture_init()

    def architecture_init(self):
        layers = []
        layers.append(BatchNorm(self.in_planes))
        layers.append(Relu())
        layers.append(Conv2D(self.in_planes, self.out_planes, 1, 1, bias_attr=bias_initializer_1x1))

        downsample = Sequential(*layers)
        return downsample

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = paddle.fluid.layers.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = paddle.fluid.layers.relu(out2)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = paddle.fluid.layers.relu(out3)
        out3 = self.conv3(out3)

        out3 = fluid.layers.concat([out1, out2, out3], 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 = fluid.layers.elementwise_add(out3, residual)
        return out3


class FAN(fluid.dygraph.Layer):
    def __init__(self, num_modules=1, end_relu=False, num_landmarks=98, fname_pretrained=None):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        # Base part
        self.conv1 = CoordConvTh(256, 256, True, False,
                                 in_channels=3, out_channels=64,
                                 kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        self.add_sublayer('m0', HourGlass(1, 4, 256, first_one=True))
        self.add_sublayer('top_m_0', ConvBlock(256, 256))
        self.add_sublayer('conv_last0', Conv2D(256, 256, 1, 1, 0))
        self.add_sublayer('bn_end0', BatchNorm(256))
        self.add_sublayer('l0', Conv2D(256, num_landmarks + 1, 1, 1, 0))

        if fname_pretrained is not None:
            self.load_pretrained_weights(fname_pretrained)

    def load_pretrained_weights(self, fname):

        model_weights, _ = fluid.dygraph.load_dygraph(fname)
        self.load_dict(model_weights)

    def forward(self, x):

        x, _ = self.conv1(x)
        x = paddle.fluid.layers.relu(self.bn1(x))

        x = fluid.layers.pool2d(self.conv2(x), pool_size=2, pool_type="avg", pool_stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        boundary_channels = []
        tmp_out = None
        ll, boundary_channel = self._sub_layers['m0'](x, tmp_out)

        ll = self._sub_layers['top_m_0'](ll)
        ll = paddle.fluid.layers.relu(self._sub_layers['bn_end0'](self._sub_layers['conv_last0'](ll)))

        # Predict heatmaps
        tmp_out = self._sub_layers['l0'](ll)
        if self.end_relu:
            tmp_out = paddle.fluid.layers.relu(tmp_out)  # HACK: Added relu
        outputs.append(tmp_out)
        boundary_channels.append(boundary_channel)
        return outputs, boundary_channels

    @fluid.dygraph.no_grad
    def get_heatmap(self, x, b_preprocess=True):
        ''' outputs 0-1 normalized heatmap '''
        import time
        st = time.time()
        x = fluid.layers.interpolate(x, out_shape=[256, 256], resample='BILINEAR')
        x_01 = x * 0.5 + 0.5

        outputs, _ = self(x_01)

        heatmaps = outputs[-1][:, :-1, :, :]

        # heatmaps = fluid.dygraph.to_variable(heatmaps)

        scale_factor = x.shape[2] // heatmaps.shape[2]
        ent = time.time()
        infer_time = (ent - st)
        # print(infer_time)
        if b_preprocess:
            # heatmaps = fluid.layers.image_resize(heatmaps, scale=scale_factor,
            #                                      resample='BILINEAR', align_corners=True)

            heatmaps = fluid.layers.interpolate(heatmaps, out_shape=[256, 256],
                                                resample='BILINEAR', align_corners=True)

            heatmaps = preprocess(heatmaps)

        return heatmaps

# ========================== #
#   Mask related functions   #
# ========================== #

def normalize(x, eps=1e-6):
    """Apply min-max normalization."""
    # x = x.contiguous()
    N, C, H, W = x.shape
    x_ = fluid.layers.reshape(x, [N * C, -1])

    max_val = fluid.layers.reduce_max(x_, dim=1, keep_dim=True)
    min_val = fluid.layers.reduce_min(x_, dim=1, keep_dim=True)

    x_ = (x_ - min_val) / (max_val - min_val + eps)

    out = fluid.layers.reshape(x_, [N, C, H, W])
    return out


def truncate(x, thres=0.1):
    """Remove small values in heatmaps."""
    temp = fluid.layers.ones_like(x) * thres
    comp = fluid.layers.less_than(x=temp, y=x).astype('float32')
    x = comp * x
    return x


def paddle_where(x, var, thres):
    temp = fluid.layers.ones_like(x) * thres
    comp = fluid.layers.less_than(x=temp, y=var).astype('float32')
    x = comp * x
    return x


def resize(x, p=2):
    """Resize heatmaps."""

    return x ** p


def shift(x, N):
    """Shift N pixels up or down."""
    up = N >= 0
    N = abs(N)
    _, _, H, W = x.shape

    head = np.arange(0, N)
    tail = np.arange(0, H - N)

    if up:
        head = np.arange(0, H - N) + N
        tail = np.arange(0, N)
    else:
        head = np.arange(0, N) + (H - N)
        tail = np.arange(0, H - N)

    # permutation indices
    perm = np.concatenate([head, tail])
    out = x[:, :, perm, :]

    return out


IDXPAIR = namedtuple('IDXPAIR', 'start end')
index_map = Munch(chin=IDXPAIR(0 + 8, 33 - 8),
                  eyebrows=IDXPAIR(33, 51),
                  eyebrowsedges=IDXPAIR(33, 46),
                  nose=IDXPAIR(51, 55),
                  nostrils=IDXPAIR(55, 60),
                  eyes=IDXPAIR(60, 76),
                  lipedges=IDXPAIR(76, 82),
                  lipupper=IDXPAIR(77, 82),
                  liplower=IDXPAIR(83, 88),
                  lipinner=IDXPAIR(88, 96))
OPPAIR = namedtuple('OPPAIR', 'shift resize')

import sys


def preprocess(x):
    """Preprocess 98-dimensional heatmaps."""
    N, C, H, W = x.shape
    x = truncate(x)
    x = normalize(x)

    sw = H // 256
    operations = Munch(chin=OPPAIR(0, 3),
                       eyebrows=OPPAIR(-7 * sw, 2),
                       nostrils=OPPAIR(8 * sw, 4),
                       lipupper=OPPAIR(-8 * sw, 4),
                       liplower=OPPAIR(8 * sw, 4),
                       lipinner=OPPAIR(-2 * sw, 3))
    x = x.numpy()
    for part, ops in operations.items():
        start, end = index_map[part]

        x[:, start:end] = resize(shift(x[:, start:end], ops.shift), ops.resize)

    zero_out = np.concatenate([np.arange(0, index_map.chin.start),
                               np.arange(index_map.chin.end, 33),
                               np.array([index_map.eyebrowsedges.start,
                                         index_map.eyebrowsedges.end,
                                         index_map.lipedges.start,
                                         index_map.lipedges.end])])
    x[:, zero_out] = 0

    start, end = index_map.nose
    x[:, start + 1:end] = shift(x[:, start + 1:end], 4 * sw)
    x[:, start:end] = resize(x[:, start:end], 1)

    start, end = index_map.eyes
    x[:, start:end] = resize(x[:, start:end], 1)
    x[:, start:end] = resize(shift(x[:, start:end], -8), 3) + \
                      shift(x[:, start:end], -24)

    # Second-level mask
    x2 = deepcopy(x)
    x2[:, index_map.chin.start:index_map.chin.end] = 0  # start:end was 0:33
    x2[:, index_map.lipedges.start:index_map.lipinner.end] = 0  # start:end was 76:96
    x2[:, index_map.eyebrows.start:index_map.eyebrows.end] = 0  # start:end was 33:51
    x = fluid.dygraph.to_variable(x)
    x2 = fluid.dygraph.to_variable(x2)
    x = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # (N, 1, H, W)
    x2 = fluid.layers.reduce_sum(x2, dim=1, keep_dim=True)  # mask without faceline and mouth
    x = x.numpy()
    x2 = x2.numpy()

    x[x != x] = 0  # set nan to zero
    x2[x != x] = 0  # set nan to zero

    x = np.clip(x, 0, 1)
    x2 = np.clip(x2, 0, 1)
    x = fluid.dygraph.to_variable(x)
    x2 = fluid.dygraph.to_variable(x2)

    return x, x2


# ========================== #
#   Mask related functions   #
# ========================== #

# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]

class HighPass(fluid.dygraph.Layer):
    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.filter = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]]).astype(np.float32) / w_hpf

    @fluid.dygraph.no_grad
    def forward(self, x):
        weight = np.expand_dims(np.expand_dims(self.filter, axis=0), axis=1).repeat(x.shape[1], axis=0)

        weight = fluid.initializer.NumpyArrayInitializer(weight)

        self.conv = Conv2D(x.shape[1], num_filters=x.shape[1], filter_size=3, padding=1, groups=x.shape[1],
                           param_attr=weight,
                           bias_attr=False)
        x = self.conv(x)
        return x


##################################################################################
# Blocks
##################################################################################

class ResBlock(fluid.dygraph.Layer):
    def __init__(self, channels_in, channels_out, normalize=False, downsample=False, use_bias=True, sn=False,
                 act=None):
        super(ResBlock, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.normalize = normalize
        self.downsample = downsample
        self.use_bias = use_bias
        self.sn = sn
        self.skip_flag = channels_in != channels_out
        self.act = act
        if self.downsample:
            self.avg_pooling0 = Pool2D(pool_size=2, pool_type="avg", pool_stride=2,
                                       pool_padding=0, global_pooling=False)
            self.avg_pooling1 = Pool2D(pool_size=2, pool_type="avg", pool_stride=2,
                                       pool_padding=0, global_pooling=False)

        self.conv1 = Conv2D(num_channels=self.channels_in, num_filters=self.channels_in,
                            filter_size=3, padding=1, param_attr=weight_initializer,
                            bias_attr=bias_initializer, stride=1, act=self.act)
        self.conv2 = Conv2D(num_channels=self.channels_in, num_filters=self.channels_out,
                            filter_size=3, padding=1, param_attr=weight_initializer,
                            bias_attr=bias_initializer, stride=1, act=self.act)
        if self.normalize:
            self.norm1 = InstanceNorm(self.channels_in)
            self.norm2 = InstanceNorm(self.channels_in)

        if self.skip_flag:
            self.conv1x1 = Conv2D(num_channels=self.channels_in, num_filters=self.channels_out,
                                  filter_size=1, padding=0, param_attr=weight_initializer,
                                  bias_attr=bias_initializer_1x1,
                                  stride=1, act=self.act)

    def shortcut(self, x):
        if self.skip_flag:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.avg_pooling0(x)
        return x

    def residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = paddle.fluid.layers.leaky_relu(x, alpha=0.2)

        x = self.conv1(x)

        if self.downsample:
            x = self.avg_pooling1(x)
        if self.normalize:
            x = self.norm2(x)
        x = paddle.fluid.layers.leaky_relu(x, alpha=0.2)
        x = self.conv2(x)

        return x

    def forward(self, x_init, training=True, mask=None):
        x = self.residual(x_init) + self.shortcut(x_init)
        return x / math.sqrt(2)  # unit variance


class AdainResBlock(fluid.dygraph.Layer):
    def __init__(self, channels_in, channels_out, upsample=False, use_bias=True, sn=False, w_hpf=0):
        super(AdainResBlock, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.upsample = upsample
        self.use_bias = use_bias
        self.sn = sn
        self.w_hpf = w_hpf
        self.skip_flag = channels_in != channels_out

        self.conv1 = Conv2D(num_channels=self.channels_in, num_filters=self.channels_out,
                            filter_size=3, padding=1, param_attr=weight_initializer,
                            bias_attr=bias_initializer, stride=1, act=None)
        self.conv2 = Conv2D(num_channels=self.channels_out, num_filters=self.channels_out,
                            filter_size=3, padding=1, param_attr=weight_initializer,
                            bias_attr=bias_initializer, stride=1, act=None)
        self.norm1 = AdaIN(self.channels_in, self.channels_in)
        self.norm2 = AdaIN(self.channels_out, self.channels_out)

        if self.skip_flag:
            self.conv1x1 = Conv2D(num_channels=self.channels_in, num_filters=self.channels_out,
                                  filter_size=1, padding=0, param_attr=weight_initializer,
                                  bias_attr=bias_initializer_1x1, stride=1, act=None)

    def shortcut(self, x):
        if self.upsample:
            x = fluid.layers.image_resize(x, scale=2, resample='NEAREST')

        if self.skip_flag:
            x = self.conv1x1(x)

        return x

    def residual(self, x, s):
        x = self.norm1([x, s])
        x = paddle.fluid.layers.leaky_relu(x, alpha=0.2)
        if self.upsample:
            x = fluid.layers.interpolate(x, scale=2, resample='NEAREST')
        x = self.conv1(x)

        x = self.norm2([x, s])
        x = paddle.fluid.layers.leaky_relu(x, alpha=0.2)
        x = self.conv2(x)

        return x

    def forward(self, x_init, training=True, mask=None):
        x_c, x_s = x_init
        x = self.residual(x_c, x_s)
        if self.w_hpf == 0:
            x = (x + self._shortcut(x)) / math.sqrt(2)
        return x


##################################################################################
# Normalization
##################################################################################


class Nop_InstanceNorm(fluid.dygraph.Layer):
    def __init__(self, num_channels, epsilon=1e-5):
        super(Nop_InstanceNorm, self).__init__()
        self.epsilon = epsilon
        self.scale = fluid.layers.fill_constant(shape=[num_channels], value=1.0, dtype='float32')
        self.bias = fluid.layers.fill_constant(shape=[num_channels], value=0.0, dtype='float32')

    def forward(self, input):
        if fluid.in_dygraph_mode():
            out, _, _ = fluid.core.ops.instance_norm(
                input, self.scale, self.bias, 'epsilon', self.epsilon)
            return out
        else:
            return fluid.layers.instance_norm(
                input,
                epsilon=self.epsilon,
                param_attr=fluid.ParamAttr(self.scale.name),
                bias_attr=fluid.ParamAttr(self.bias.name))


class AdaIN(fluid.dygraph.Layer):
    def __init__(self, input_dim, channels, sn=False, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.channels = channels
        self.epsilon = epsilon
        self.input_dim = input_dim
        self.norm = Nop_InstanceNorm(self.input_dim)
        self.fc = Linear(64, self.input_dim * 2,
                         param_attr=weight_initializer,
                         act=None)

    def forward(self, x_init, training=True, mask=None):
        x, style = x_init
        x_norm = self.norm(x)
        h = self.fc(style)
        h = fluid.layers.unstack(h, axis=1)

        gamma = fluid.layers.stack(h[:self.input_dim], axis=1)
        beta = fluid.layers.stack(h[-self.input_dim:], axis=1)

        gamma = fluid.layers.reshape(gamma, shape=[-1, self.channels, 1, 1])
        beta = fluid.layers.reshape(beta, shape=[-1, self.channels, 1, 1])
        gamma = fluid.layers.ones_like(gamma) + gamma
        x = fluid.layers.elementwise_mul(x_norm, gamma)
        x = fluid.layers.elementwise_add(x, beta)
        return x


##################################################################################
# Activation Function
##################################################################################

class Leaky_Relu(fluid.dygraph.Layer):
    def __init__(self, alpha=0.2):
        super(Leaky_Relu, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return paddle.fluid.layers.leaky_relu(x, self.alpha)


class Relu(fluid.dygraph.Layer):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        return paddle.fluid.layers.relu(x)

##################################################################################
# GAN Loss Function
##################################################################################

# def regularization_loss(model):
#    loss = tf.nn.scale_regularization_loss(model.losses)
#
# return loss


def L1_loss(x, y):
    l1_loss = fluid.dygraph.L1Loss(reduction='mean')
    loss = l1_loss(x, y)

    return loss


def squared_difference(x, y):
    net_0 = fluid.layers.elementwise_sub(x, y)
    net_1 = fluid.layers.elementwise_mul(net_0, net_0)
    return net_1


def discriminator_loss(gan_type, real_logit, fake_logit):
    real_loss = 0
    fake_loss = 0

    if gan_type == 'lsgan':
        real_loss = fluid.layers.mean(squared_difference(real_logit, fluid.layers.ones_like(real_logit)))
        fake_loss = fluid.layers.mean(paddle.fluid.layers.square(fake_logit))

    if gan_type == 'gan' or gan_type == 'gan-gp':
        real_loss = fluid.layers.mean(
            fluid.layers.sigmoid_cross_entropy_with_logits(real_logit, fluid.layers.ones_like(real_logit)))

        fake_loss = fluid.layers.mean(
            fluid.layers.sigmoid_cross_entropy_with_logits(fake_logit, fluid.layers.zeros_like(fake_logit)))

    if gan_type == 'hinge':
        real_loss = fluid.layers.mean(Relu(1.0 - real_logit))
        fake_loss = fluid.layers.meann(Relu(1.0 + fake_logit))

    return real_loss, fake_loss


def generator_loss(gan_type, fake_logit):
    fake_loss = 0

    if gan_type == 'lsgan':
        fake_loss = fluid.layers.mean(squared_difference(fake_logit, fluid.layers.ones_like(fake_logit)))
    if gan_type == 'gan' or gan_type == 'gan-gp':
        fake_loss = fluid.layers.mean(
            fluid.layers.sigmoid_cross_entropy_with_logits(fake_logit, paddle.fluid.layers.ones_like(fake_logit)))

    if gan_type == 'hinge':
        fake_loss = -fluid.layers.mean(fake_logit)

    return fake_loss


def cal_gradient_penalty(netD, real_data, fake_data, edge_data=None, type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = paddle.rand((real_data.shape[0], 1))
            alpha = paddle.expand(alpha, [1, np.prod(real_data.shape) // real_data.shape[0]])
            alpha = paddle.reshape(alpha, real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        # interpolatesv.requires_grad_(True)
        interpolatesv.stop_gradient = False
        real_data.stop_gradient = True
        fake_AB = paddle.concat((real_data.detach(), interpolatesv), 1)
        disc_interpolates = netD(fake_AB)

        # FIXME: use paddle.ones
        outs = paddle.fill_constant(disc_interpolates.shape, disc_interpolates.dtype, 1.0)
        gradients = paddle.imperative.grad(outputs=disc_interpolates, inputs=fake_AB,
                                           grad_outputs=outs,  # paddle.ones(list(disc_interpolates.shape)),
                                           create_graph=True,
                                           retain_graph=True,
                                           only_inputs=True,
                                           # no_grad_vars=set(netD.parameters())
                                           )

        gradients = paddle.reshape(gradients[0], [real_data.shape[0], -1])  # flat the data

        gradient_penalty = paddle.reduce_mean(
            (paddle.norm(gradients + 1e-16, 2, 1) - constant) ** 2) * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def r1_gp_req(discriminator, x_real, y_org, create_graph=True):
    x_real.stop_gradient = False
    real_loss = discriminator([x_real, y_org])
    real_loss = fluid.layers.reduce_sum(real_loss)
    real_grads = fluid.dygraph.grad(
        outputs=real_loss,
        inputs=x_real,
        no_grad_vars=None,
        create_graph=False)[0]

    assert (real_grads.shape == x_real.shape)

    grad_shape = real_grads.shape
    grad = fluid.layers.reshape(
        real_grads, [-1, grad_shape[1] * grad_shape[2] * grad_shape[3]])
    grad = fluid.layers.reduce_sum(fluid.layers.square(grad), dim=1)
    r1_penalty = 0.5 * fluid.layers.reduce_mean(grad)
    return r1_penalty

