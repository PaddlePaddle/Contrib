from modules.util import AntiAliasInterpolation2d, make_coordinate_grid

import paddle
import logging
import numpy as np
from paddle import fluid
from paddle.fluid import dygraph

PP_v2 = True if paddle.version.major == '2' else False

logging.warning('Need second order derivative model.py:L170-258')
TEST_MODE = False
if TEST_MODE:
    logging.warning('TEST MODE: model.py')


class conv_block(dygraph.Layer):
    def __init__(self, input_channels, num_filter, groups, name=None, use_bias=False):
        super(conv_block, self).__init__()
        self._layers = []
        i = 0
        self.conv_in = dygraph.Conv2D(
            num_channels=input_channels,
            num_filters=num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                name=name + str(i + 1) + "_weights"),
            bias_attr=False if not use_bias else fluid.param_attr.ParamAttr(
                name=name + str(i + 1) + "_bias"))
        if groups == 1:
            return
        for i in range(1, groups):
            _a = dygraph.Conv2D(
                num_channels=num_filter,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_weights"),
                bias_attr=False if not use_bias else fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_bias"))
            self._layers.append(_a)
        self.conv = dygraph.Sequential(*self._layers)
    
    def forward(self, x):
        feat = self.conv_in(x)
        out = fluid.layers.pool2d(input=self.conv(feat), pool_size=2, pool_type='max', pool_stride=2)
        return out, feat


class Vgg19(dygraph.Layer):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    
    def __init__(self, layers=19, class_dim=1000, torch_version=True, requires_grad=False):
        super(Vgg19, self).__init__()
        self.layers = layers
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        assert layers in vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)
        
        nums = vgg_spec[layers]
        self.conv1 = conv_block(3, 64, nums[0], name="conv1_", use_bias=True if torch_version else False)
        self.conv2 = conv_block(64, 128, nums[1], name="conv2_", use_bias=True if torch_version else False)
        self.conv3 = conv_block(128, 256, nums[2], name="conv3_", use_bias=True if torch_version else False)
        self.conv4 = conv_block(256, 512, nums[3], name="conv4_", use_bias=True if torch_version else False)
        self.conv5 = conv_block(512, 512, nums[4], name="conv5_", use_bias=True if torch_version else False)
        _a = fluid.ParamAttr(
            initializer=fluid.initializer.NumpyArrayInitializer(np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)),
            trainable=False)
        self.mean = self.create_parameter(shape=(1, 3, 1, 1), attr=_a, dtype="float32")
        _a = fluid.ParamAttr(
            initializer=fluid.initializer.NumpyArrayInitializer(np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)),
            trainable=False)
        self.std = self.create_parameter(shape=(1, 3, 1, 1), attr=_a, dtype="float32")
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = True
    
    def forward(self, x):
        x = (x - fluid.layers.expand_as(self.mean, x)) / fluid.layers.expand_as(self.std, x)
        feat, feat_1 = self.conv1(x)
        feat, feat_2 = self.conv2(feat)
        feat, feat_3 = self.conv3(feat)
        feat, feat_4 = self.conv4(feat)
        _, feat_5 = self.conv5(feat)
        return [feat_1, feat_2, feat_3, feat_4, feat_5]


class ImagePyramide(dygraph.Layer):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        self.downs = dygraph.LayerList()
        self.name_list = []
        for scale in scales:
            self.downs.add_sublayer(str(scale).replace('.', '-'), AntiAliasInterpolation2d(num_channels, scale))
            self.name_list.append(str(scale).replace('.', '-'))
    
    def forward(self, x):
        out_dict = {}
        for scale, down_module in zip(self.name_list, self.downs):
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    
    def __init__(self, bs, **kwargs):
        # noise = np.random.normal(loc=0, scale=kwargs['sigma_affine'], size=(bs, 2, 3))
        noise = fluid.layers.Normal(loc=[0], scale=[kwargs['sigma_affine']]).sample([bs, 2, 3])
        noise = fluid.layers.reshape(noise, (bs, 2, 3))
        if TEST_MODE:
            logging.warning('TEST MODE: Transform.noise == np.ones model.py:L135')
            noise = dygraph.to_variable(np.ones((bs, 2, 3)).astype(np.float32))
        
        self.theta = noise + fluid.layers.reshape(fluid.layers.eye(2, 3), (1, 2, 3))
        self.bs = bs
        
        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), 'float32')
            self.control_points = fluid.layers.unsqueeze(self.control_points, [0])
            if TEST_MODE:
                logging.warning('TEST MODE: Transform.control_params == np.ones model.py:L144')
                self.control_params = dygraph.to_variable(np.ones((bs, 1, kwargs['points_tps'] ** 2)))
            else:
                buf = fluid.layers.Normal(loc=[0], scale=[kwargs['sigma_tps']]).sample(
                    [bs, 1, kwargs['points_tps'] ** 2])
                self.control_params = fluid.layers.reshape(buf, (bs, 1, kwargs['points_tps'] ** 2))
                # self.control_params = dygraph.to_variable(
                #     np.random.normal(loc=0, scale=kwargs['sigma_tps'], size=(bs, 1, kwargs['points_tps'] ** 2)))
        else:
            self.tps = False
    
    def transform_frame(self, frame):
        grid = fluid.layers.unsqueeze(make_coordinate_grid(frame.shape[2:], 'float32'), [0])
        grid = fluid.layers.reshape(grid, (1, frame.shape[2] * frame.shape[3], 2))
        grid = fluid.layers.reshape(self.warp_coordinates(grid), (self.bs, frame.shape[2], frame.shape[3], 2))
        if TEST_MODE:
            bf = fluid.layers.grid_sampler(frame, grid)
            logging.warning('TEST MODE Output of fluid.layers.grid_sampler == 2. model:L152')
            return fluid.dygraph.to_variable(np.ones(bf.shape).astype(np.float32) * 2)
        # 0.0.0c 分支等待更新
        elif PP_v2:
            # return fluid.layers.grid_sampler(frame, grid)
            return fluid.layers.grid_sampler(frame, grid, mode='bilinear', padding_mode='reflect', align_corners=False)
        else:
            return fluid.layers.grid_sampler(frame, grid)
    
    ## 使用grad()的正常版本（等待二阶更新）
    # def warp_coordinates(self, coordinates):
    #     # self.theta is float32
    #     theta = self.theta.astype('float32')
    #     theta = fluid.layers.unsqueeze(theta, 1)
    #     coordinates = fluid.layers.unsqueeze(coordinates, -1)
    #     # If x1:(1, 5, 2, 2), x2:(10, 100, 2, 1)
    #     # torch.matmul can broadcast x1, x2 to (10, 100, ...)
    #     # In PDPD, it should be done manually
    #     theta_part_a = theta[:, :, :, :2]
    #     theta_part_b = theta[:, :, :, 2:]
    #     # Use broadcast_v1 instead
    #     transformed = fluid.layers.matmul(*broadcast_v1(theta_part_a, coordinates)) + theta_part_b
    #     transformed = fluid.layers.squeeze(transformed, [-1])
    #     if self.tps:
    #         control_points = self.control_points.astype('float32')
    #         control_params = self.control_params.astype('float32')
    #         distances = fluid.layers.reshape(coordinates, (coordinates.shape[0], -1, 1, 2)) - fluid.layers.reshape(
    #             control_points, (1, 1, -1, 2))
    #         distances = fluid.layers.reduce_sum(fluid.layers.abs(distances), -1)
    
    #         result = distances ** 2
    #         result = result * fluid.layers.log(distances + 1e-6)
    #         result = result * control_params
    #         result = fluid.layers.reshape(fluid.layers.reduce_sum(result, 2), (self.bs, coordinates.shape[1], 1))
    #         transformed = transformed + result
    #     return transformed
    
    # def jacobian(self, coordinates):
    #     new_coordinates = self.warp_coordinates(coordinates)  # When batch_size is 5, the shape of coordinates and new_coordinates is (5, 10, 2)
    #     # PDPD cannot use new_coordinates[..., 0]
    #     assert len(new_coordinates.shape) == 3
    #     grad_x = dygraph.grad(fluid.layers.reduce_sum(new_coordinates[:, :, 0]), coordinates, create_graph=True)
    #     grad_y = dygraph.grad(fluid.layers.reduce_sum(new_coordinates[:, :, 1]), coordinates, create_graph=True)
    #     jacobian = fluid.layers.concat([fluid.layers.unsqueeze(grad_x[0], -2), fluid.layers.unsqueeze(grad_y[0], -2)],
    #                                    axis=-2)
    #     return jacobian
    
    ## 手算一阶导的魔改版本
    def warp_coordinates(self, coordinates_in, need_grad=False):
        in_shape = coordinates_in.shape
        theta = self.theta.astype('float32')
        theta = fluid.layers.reshape(theta, (theta.shape[0], 1, theta.shape[1], theta.shape[2]))
        coordinates = fluid.layers.reshape(coordinates_in, (*coordinates_in.shape, 1))
        theta_parta = theta[:, :, :, :2]
        transformed = fluid.layers.matmul(*broadcast_v1(theta_parta, coordinates)) + theta[:, :, :, 2:]
        transformed = fluid.layers.squeeze(transformed, [-1])
        if self.tps:
            control_points = self.control_points.astype('float32')
            control_params = self.control_params.astype('float32')
            _a = fluid.layers.reshape(coordinates, (coordinates.shape[0], -1, 1, 2))
            distances_0 = _a - fluid.layers.reshape(control_points, (1, 1, -1, 2))
            distances_1 = fluid.layers.abs(distances_0)
            distances = fluid.layers.reduce_mean(distances_1, -1) * fluid.dygraph.to_variable(
                np.array([distances_1.shape[-1]]).astype(np.float32))
            result0 = distances * distances
            result1 = result0 * fluid.layers.log(distances + 1e-6)
            result2 = result1 * control_params
            result3 = fluid.layers.reshape(fluid.layers.reduce_mean(result2, 2) * fluid.dygraph.to_variable(np.array([result2.shape[2]]).astype(np.float32)), (self.bs, coordinates.shape[1], 1))
            transformed = transformed + result3
        if need_grad:
            _theta_part_a, _ = broadcast_v1(theta_parta, coordinates)
            _buf = _theta_part_a
            Dtransformed_0_Din = fluid.layers.unsqueeze(fluid.layers.reduce_mean(_buf[:, :, 0, :], [0]), [0])
            Dtransformed_1_Din = fluid.layers.unsqueeze(fluid.layers.reduce_mean(_buf[:, :, 1, :], [0]), [0])
            if self.tps:
                bool_mat = distances_0 > fluid.dygraph.to_variable(np.array([0]).astype(np.float32))
                A = (fluid.dygraph.grad(result3, result1, create_graph=True)[0] * (fluid.dygraph.grad(fluid.layers.reduce_mean(result0) * fluid.dygraph.to_variable(np.prod(result0.shape).astype(np.float32).reshape(1)), distances, create_graph=True)[0] * fluid.layers.log(distances + 1e-6) + result0 * 1 / (distances + 1e-6)))
                A2 = fluid.layers.unsqueeze(A, [-1])
                A3 = A2 * (bool_mat.astype('float32') * 2 - 1)
                A4 = fluid.layers.reduce_sum(A3, -2)
                A5 = fluid.layers.reshape(A4, in_shape)
                return transformed, (Dtransformed_0_Din + A5, Dtransformed_1_Din + A5)
            else:
                return transformed, (Dtransformed_0_Din, Dtransformed_1_Din)
        return transformed
    
    def jacobian(self, coordinates):
        new_coordinates, (grad_x, grad_y) = self.warp_coordinates(coordinates, True)
        assert len(new_coordinates.shape) == 3
        jacobian = fluid.layers.concat([fluid.layers.unsqueeze(grad_x, [-2]), fluid.layers.unsqueeze(grad_y, [-2])], axis=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(dygraph.Layer):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """
    
    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        
        self.loss_weights = train_params['loss_weights']
        
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
    
    def forward(self, x):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])
        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        
        loss_values = {}
        
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        
        # VGG19 perceptual Loss
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = fluid.layers.reduce_mean(fluid.layers.abs(x_vgg[i] - y_vgg[i].detach()))
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total
        
        # Generator Loss
        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = fluid.layers.reduce_mean((1 - discriminator_maps_generated[key]) ** 2)
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total
            
            # Feature matching Loss
            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = fluid.layers.reduce_mean(fluid.layers.abs(a - b))
                        value_total += self.loss_weights['feature_matching'][i] * value
                loss_values['feature_matching'] = value_total
        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)
            
            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp
            
            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = fluid.layers.reduce_mean(
                    fluid.layers.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])))
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value
            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = fluid.layers.matmul(
                    *broadcast_v1(transform.jacobian(transformed_kp['value']), transformed_kp['jacobian']))
                if PP_v2:
                    normed_driving = paddle.inverse(kp_driving['jacobian'])
                else:
                    normed_driving = _inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = fluid.layers.matmul(*broadcast_v1(normed_driving, normed_transformed))
                eye = dygraph.to_variable(fluid.layers.reshape(fluid.layers.eye(2, 2, dtype='float32'), (1, 1, 2, 2)))
                
                value = fluid.layers.reduce_mean(fluid.layers.abs(eye - value))
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value
        return loss_values, generated


class DiscriminatorFullModel(dygraph.Layer):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """
    
    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        self.loss_weights = train_params['loss_weights']
    
    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())
        
        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
        
        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * fluid.layers.reduce_mean(value)
        loss_values['disc_gan'] = value_total
        
        return loss_values


def broadcast_v1(x, y):
    """
    Broadcast before matmul
    """
    if len(x.shape) != len(y.shape):
        print(x.shape, '!=', y.shape)
        raise ValueError()
    *dim_x, _, _ = x.shape
    *dim_y, _, _ = y.shape
    max_shape = np.max(np.stack([dim_x, dim_y], axis=0), axis=0)
    if np.count_nonzero(max_shape % np.array(dim_x)) != 0 or np.count_nonzero(max_shape % np.array(dim_y)) != 0:
        raise ValueError()
    x_bc = fluid.layers.expand(x, (*((max_shape / np.array(dim_x)).astype(np.int32).tolist()), 1, 1)).astype('float32')
    y_bc = fluid.layers.expand(y, (*((max_shape / np.array(dim_y)).astype(np.int32).tolist()), 1, 1)).astype('float32')
    return x_bc, y_bc

def _inverse(x):
    """手动求逆以兼容1.8.x集群
    """
    assert x.shape[-1] == 2 and x.shape[-2] == 2
    buf = fluid.layers.reshape(x, (-1, *(x.shape[-2:])))
    a, b, c, d = buf[:, 0, 0], buf[:, 1, 1], buf[:, 0, 1], buf[:, 1, 0]
    div = fluid.layers.reshape(a * b - c * d, (-1, 1, 1))
    n_a, n_b, n_c, n_d = b, a, -c, -d
    colum_0 = fluid.layers.stack([n_a, n_d], axis=-1)
    colum_1 = fluid.layers.stack([n_c, n_b], axis=-1)
    mat = fluid.layers.stack([colum_0, colum_1], axis=-1) / div
    return fluid.layers.reshape(mat, x.shape)
