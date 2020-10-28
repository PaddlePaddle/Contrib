import logging

import paddle

if paddle.version.major == '2':
    PP_v2 = True
    from paddle.nn import functional as F
else:
    PP_v2 = False
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork
from paddle import fluid
from paddle.fluid import dygraph

TEST_MODE = False
if TEST_MODE:
    logging.warning('TEST MODE: Output of fluid.layers.grid_sampler is 2. generator:L83')


# TODO: test OcclusionAwareGenerator
# Programing......None

class OcclusionAwareGenerator(dygraph.Layer):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """
    
    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()
        
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None
        
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = dygraph.LayerList(down_blocks)
        
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = dygraph.LayerList(up_blocks)
        
        self.bottleneck = dygraph.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_sublayer('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        
        self.final = dygraph.Conv2D(block_expansion, num_channels, filter_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
    
    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = fluid.layers.transpose(deformation, (0, 3, 1, 2))
            if PP_v2:
                deformation = F.interpolate(deformation, size=(h, w), mode='BILINEAR', align_corners=False)
            else:
                deformation = fluid.layers.interpolate(deformation, out_shape=(h, w), resample='BILINEAR')
            deformation = fluid.layers.transpose(deformation, (0, 2, 3, 1))
        if TEST_MODE:
            import numpy as np
            bf = fluid.layers.grid_sampler(inp, deformation)
            return fluid.dygraph.to_variable(np.ones(bf.shape).astype(np.float32) * 2)
        elif PP_v2:
            # return fluid.layers.grid_sampler(inp, deformation)
            return fluid.layers.grid_sampler(inp, deformation, mode='bilinear', padding_mode='zeros', align_corners=False)
        else:
            return fluid.layers.grid_sampler(inp, deformation)
    
    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving, kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']
            
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)
            
            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    if PP_v2:
                        occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='BILINEAR', align_corners=False)
                    else:
                        occlusion_map = fluid.layers.interpolate(occlusion_map, out_shape=out.shape[2:], resample='BILINEAR')
                out = out * occlusion_map
            output_dict["deformed"] = self.deform_input(source_image, deformation)
        
        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = fluid.layers.sigmoid(out)
        output_dict["prediction"] = out
        return output_dict
