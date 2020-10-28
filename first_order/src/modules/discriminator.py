from paddle import fluid
from paddle.fluid import dygraph

from modules.util import kp2gaussian


class DownBlock2d(dygraph.Layer):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = dygraph.Conv2D(in_features, out_features, filter_size=kernel_size)

        if sn:
            self.sn = dygraph.SpectralNorm(self.conv.weight.shape, dim=0)
        else:
            self.sn = None
        if norm:
            self.norm = dygraph.InstanceNorm(num_channels=out_features, epsilon=1e-05, dtype='float32')
        else:
            self.norm = None

        self.pool = pool

    def forward(self, x):
        out = x
        if self.sn is not None:
            self.conv.weight.set_value(self.sn(self.conv.weight))
        out = self.conv(out)
        if self.norm is not None:
            out = self.norm(out)
        out = fluid.layers.leaky_relu(out, 0.2)
        if self.pool:
            out = fluid.layers.pool2d(out, pool_size=2, pool_type='avg', pool_stride=2, ceil_mode=False)
        return out


class Discriminator(dygraph.Layer):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = dygraph.LayerList(down_blocks)
        self.conv = dygraph.Conv2D(self.down_blocks[len(self.down_blocks) - 1].conv.parameters()[0].shape[0], 1, filter_size=1)
        if sn:
            self.sn = dygraph.SpectralNorm(self.conv.parameters()[0].shape, dim=0)
        else:
            self.sn = None
        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            out = fluid.layers.concat([out, heatmap], axis=1)
        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        if self.sn is not None:
            self.conv.weight.set_value(self.sn(self.conv.parameters()[0]))
        prediction_map = self.conv(out)
        return feature_maps, prediction_map


class MultiScaleDiscriminator(dygraph.Layer):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        self.discs = dygraph.LayerList()
        self.nameList = []
        for scale in scales:
            self.discs.add_sublayer(str(scale).replace('.', '-'), Discriminator(**kwargs))
            self.nameList.append(str(scale).replace('.', '-'))

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in zip(self.nameList, self.discs):
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            feature_maps, prediction_map = disc(x[key], kp)
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict
