import numpy as np
from paddle import fluid
from paddle.fluid import dygraph

from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid_cpu


class KPDetector(dygraph.Layer):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = dygraph.Conv2D(num_channels=self.predictor.out_filters, num_filters=num_kp, filter_size=(7, 7),
                                 padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = dygraph.Conv2D(num_channels=self.predictor.out_filters,
                                           num_filters=4 * self.num_jacobian_maps, filter_size=(7, 7), padding=pad
                                           )
            self.jacobian.weight.set_value(np.zeros(list(self.jacobian.weight.shape), dtype=np.float32))
            self.jacobian.bias.set_value(np.array([1, 0, 0, 1] * self.num_jacobian_maps, dtype=np.float32))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap) -> dict:
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = fluid.layers.unsqueeze(heatmap, [-1])
        grid = make_coordinate_grid_cpu(shape[2:], np.float32)[np.newaxis, np.newaxis, ...]
        grid = dygraph.to_variable(grid)
        value = fluid.layers.reduce_sum(heatmap * grid, [2, 3])
        kp = {'value': value}
        return kp

    def forward(self, x) -> dict:
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)  # type: paddle.fluid.core_avx.VarBase
        prediction = self.kp(feature_map)   # type: paddle.fluid.core_avx.VarBase
        final_shape = prediction.shape
        
        heatmap = fluid.layers.reshape(prediction, (final_shape[0], final_shape[1], -1))
        heatmap = fluid.layers.softmax(heatmap / self.temperature, axis=2)
        heatmap = fluid.layers.reshape(heatmap, final_shape)
        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = fluid.layers.reshape(jacobian_map,
                                                (final_shape[0], self.num_jacobian_maps, 4, final_shape[2], final_shape[3]))
            heatmap = fluid.layers.unsqueeze(heatmap, [2])
            
            jacobian = heatmap * jacobian_map
            jacobian = fluid.layers.reshape(jacobian, (final_shape[0], final_shape[1], 4, -1))
            jacobian = fluid.layers.reduce_sum(jacobian, dim=-1)
            jacobian = fluid.layers.reshape(jacobian, (jacobian.shape[0], jacobian.shape[1], 2, 2))
            out['jacobian'] = jacobian
        return out
