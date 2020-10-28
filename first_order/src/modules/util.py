import paddle
if paddle.version.major == '2':
    PP_v2 = True
    from paddle.nn import functional as F
else:
    PP_v2 = False
import numpy as np
from paddle import fluid
from paddle.fluid import dygraph


def kp2gaussian(kp, spatial_size, kp_variance: np.ndarray) -> np.ndarray:
    """
    Transform a keypoint into gaussian like representation
    BP is supported
    """
    if isinstance(kp['value'], fluid.core_avx.VarBase):
        mean = kp['value'].numpy()
    elif isinstance(kp['value'], np.ndarray):
        mean = kp['value']
    else:
        raise TypeError('TYPE of keypoint : %s is not supported' % type(kp['value']))

    coordinate_grid = make_coordinate_grid_cpu(spatial_size, mean.dtype)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.reshape(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = np.tile(coordinate_grid, repeats)
    coordinate_grid = dygraph.to_variable(coordinate_grid)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = fluid.layers.reshape(kp['value'], shape)

    mean_sub = (coordinate_grid - mean)
    if isinstance(kp_variance, fluid.core_avx.VarBase):
        pass
    elif isinstance(kp_variance, (np.ndarray, float)):
        kp_variance = dygraph.to_variable(np.array([kp_variance]).astype(np.float32))
    else:
        raise TypeError('TYPE of keypoint : %s is not supported' % type(kp_variance))
    out = fluid.layers.exp(-0.5 * fluid.layers.reduce_sum((mean_sub ** 2), -1) / kp_variance)

    return out


make_coordinate_grid_cpu = lambda spatial_size, ttype: np.stack(
    np.meshgrid(np.linspace(-1, 1, spatial_size[1]), np.linspace(-1, 1, spatial_size[0])), axis=-1).astype(np.float32)
make_coordinate_grid = lambda spatial_size, ttype: dygraph.to_variable(
    make_coordinate_grid_cpu(spatial_size, np.float32)).astype(ttype)


######################################################################
# def make_coordinate_grid(spatial_size, type):
#     """
#     Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
#     """
#     h, w = spatial_size
#     x = torch.arange(w).type(type)
#     y = torch.arange(h).type(type)
#
#     x = (2 * (x / (w - 1)) - 1)
#     y = (2 * (y / (h - 1)) - 1)
#
#     yy = y.view(-1, 1).repeat(1, w)
#     xx = x.view(1, -1).repeat(h, 1)
#
#     meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
######################################################################

class ResBlock2d(dygraph.Layer):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding, **kwargs):
        super(ResBlock2d, self).__init__(**kwargs)
        self.conv1 = dygraph.Conv2D(num_channels=in_features, num_filters=in_features, filter_size=kernel_size, padding=padding)
        self.conv2 = dygraph.Conv2D(num_channels=in_features, num_filters=in_features, filter_size=kernel_size, padding=padding)
        self.norm1 = dygraph.BatchNorm(num_channels=in_features, momentum=0.1)
        self.norm2 = dygraph.BatchNorm(num_channels=in_features, momentum=0.1)

    def forward(self, x):
        out = self.norm1(x)
        out = fluid.layers.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = fluid.layers.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(dygraph.Layer):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        self.conv = dygraph.Conv2D(
            num_channels=in_features,
            num_filters=out_features,
            filter_size=kernel_size,
            padding=padding,
            groups=groups)
        self.norm = dygraph.BatchNorm(num_channels=out_features, momentum=0.1)

    def forward(self, x):
        if PP_v2:
            out = F.interpolate(x, scale_factor=2, mode='NEAREST', align_corners=False)
        else:
            out = fluid.layers.interpolate(x, scale=2, resample='NEAREST')
        out = self.conv(out)
        out = self.norm(out)
        out = fluid.layers.relu(out)
        return out


class DownBlock2d(dygraph.Layer):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = dygraph.Conv2D(num_channels=in_features, num_filters=out_features, filter_size=kernel_size, padding=padding, groups=groups)
        self.norm = dygraph.BatchNorm(num_channels=out_features, momentum=0.1)
        self.pool = dygraph.Pool2D(pool_size=(2, 2), pool_type='avg', pool_stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = fluid.layers.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(dygraph.Layer):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = dygraph.Conv2D(
            num_channels=in_features, num_filters=out_features, filter_size=kernel_size, padding=padding, groups=groups)
        self.norm = dygraph.BatchNorm(out_features)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = fluid.layers.relu(out)
        return out


class Encoder(dygraph.Layer):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = dygraph.LayerList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(dygraph.Layer):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = dygraph.LayerList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            # TODO: If the size of width or length is odd, out and skip cannot concat
            out = fluid.layers.concat([out, skip], axis=1)
        return out


class Hourglass(dygraph.Layer):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x) -> paddle.fluid.core_avx.VarBase:
        return self.decoder(self.encoder(x))


# TODO: 20200810
class AntiAliasInterpolation2d(dygraph.Layer):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        # TODO: kernel DO NOT NEED BP, initialized in cpu by numpy
        meshgrids = np.meshgrid(
            *[
                np.arange(size, dtype=np.float32)
                for size in kernel_size
            ]
        )
        meshgrids = [i.T for i in meshgrids]
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= np.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.reshape(1, 1, *kernel.shape)
        kernel = kernel.repeat(channels, 0)  # [1, 1, *kernel.shape] -> [channels, 1, *kernel.shape]
        self.kernel_attr = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(kernel), trainable=False)
        self.kernel = self.create_parameter(kernel.shape, attr=self.kernel_attr, dtype="float32")
        self.groups = channels
        self.scale = scale
        self.conv = dygraph.Conv2D(channels, channels, filter_size=kernel.shape[-1], groups=self.groups,
                                   param_attr=self.kernel_attr, bias_attr=False)
        self.conv.weight.set_value(kernel)

    def forward(self, input):
        if self.scale == 1.0:
            return input
        
        out = fluid.layers.pad2d(input=input, paddings=[self.ka, self.kb, self.ka, self.kb], mode='constant')
        out = self.conv(out)
        # TODO: fluid.layers.interpolate IS NOT SAME WITH F.interpolate due to align_corners==True, use fluid.layers.resize_nearest instead.
        if PP_v2:
            out = F.interpolate(out, scale_factor=self.scale, mode='NEAREST', align_corners=False)
        else:
            out = fluid.layers.resize_nearest(out, scale=self.scale, align_corners=False)
        return out
