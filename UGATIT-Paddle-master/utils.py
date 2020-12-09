import os
import cv2
import numpy as np
from paddle.fluid.layers import sigmoid_cross_entropy_with_logits, reduce_sum, reduce_mean, clip, pad2d, relu, \
    leaky_relu, tanh, interpolate
from paddle.fluid.dygraph import SpectralNorm,Layer

""" tools"""


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


def denorm(x):
    return x * 0.5 + 0.5


def tensor2numpy(x):
    return x.detach().numpy().transpose(1, 2, 0)


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def cam(x, size=256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0


"""some api"""


class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = reduction

    def __call__(self, x, label):
        out = sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return reduce_sum(out)
        elif self.reduction == 'mean':
            return reduce_mean(out)
        else:
            return out


class RhoClipper():
    def __init__(self, vmin=0, vmax=1):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, net):
        for name, param in net.named_parameters():
            if 'rho' in name:
                param.set_value(clip(param, self.vmin, self.vmax))


class ReflectionPad2D(Layer):
    def __init__(self, paddings):
        super().__init__()
        self.padding = [paddings] * 4

    def forward(self, x):
        return pad2d(x, self.padding, mode='reflect')


class ReLU(Layer):
    def __init__(self, inplace=True):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.set_value(relu(x))
            return x
        else:
            y = relu(x)
            return y


class LeakyReLU(Layer):
    def __init__(self, alpha=0.02, inplce=False):
        super(LeakyReLU, self).__init__()
        self.inplce = inplce
        self.alpha = alpha

    def forward(self, x):
        if self.inplce:
            x.set_value(leaky_relu(x, self.alpha))
            return x
        else:
            y = leaky_relu(x, self.alpha)
            return y


class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return tanh(x)


class Upsample(Layer):
    def __init__(self, scales, resamples):
        super(Upsample, self).__init__()
        self.scale = scales
        self.resample = resamples

    def forward(self, x):
        return interpolate(x, scale=self.scale, resample=self.resample)


def var(input, dim=None, keep_dim=True, unbiased=True, name=None):
    rank = len(input.shape)
    dims = dim if dim is not None and dim != [] else range(rank)
    dims = [e if e >= 0 else e + rank for e in dims]
    inp_shape = input.shape
    mean = reduce_mean(input, dim=dim, keep_dim=True, name=name)
    tmp = reduce_mean((input - mean) ** 2, dim=dim, keep_dim=True, name=name)
    if unbiased:
        n = 1
        for i in dims:
            n *= inp_shape[i]
        factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    return tmp


class spectral_norm(Layer):

    def __init__(self, layer, dim=0, power_iters=1, eps=1e-12, dtype='float32'):
        super(spectral_norm, self).__init__()
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        self.dtype = dtype
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.shape = weight.shape
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = SpectralNorm(self.shape, self.dim, self.power_iters, self.eps, self.dtype)(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out
