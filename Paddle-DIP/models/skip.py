import paddle
import paddle.nn as nn
from .common import *

class Down(nn.Layer):
    def __init__(self, n_input, n_down, filter_size_down,
                 need_bias=True, act_fun='LeakyReLU',
                 pad='zero', downsample_mode='stride'):
        super(Down, self).__init__()
        self.conv1 = Conv(n_input, n_down, filter_size_down, stride=2, bias=need_bias,
                          act_fun=act_fun, pad=pad, downsample_mode=downsample_mode)
        self.conv2 = Conv(n_down, n_down, filter_size_down, stride=1, bias=need_bias,
                          act_fun=act_fun, pad=pad)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Up(nn.Layer):
    def __init__(self, n_input, n_up, filter_size_up,
                 need_up=True, need_bias=True,
                 act_fun='LeakyReLU', pad='zero'):
        super(Up, self).__init__()
        self.bn = nn.BatchNorm2D(n_input)
        self.conv1 = Conv(n_input, n_up, filter_size_up, stride=1, bias=need_bias,
                          act_fun=act_fun, pad=pad)
        self.need_up = need_up
        if self.need_up:
            self.conv2 = Conv(n_up, n_up, 1, stride=1, bias=need_bias,
                              act_fun=act_fun, pad=pad)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        if self.need_up:
            x = self.conv2(x)
        return x


class Concat(nn.Layer):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return paddle.concat(inputs_, axis=self.dim)



class SkipNet(nn.Layer):
    def __init__(self, num_input_channels=2, num_output_channels=3,
                 num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4],
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                 need_sigmoid=True, need_bias=True,
                 pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
                 need1x1_up=True):

        """Assembles encoder-decoder with skip connections.

        Arguments:
            act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
            pad (string): zero|reflection (default: 'zero')
            upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
            downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

        """
        super(SkipNet, self).__init__()

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        n_scales = len(num_channels_down)

        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
            upsample_mode = [upsample_mode] * n_scales

        if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
            downsample_mode = [downsample_mode] * n_scales

        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
            filter_size_down = [filter_size_down] * n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
            filter_size_up = [filter_size_up] * n_scales

        self.n_scales = n_scales
        self.Down_path = nn.LayerList()
        self.Up_path1 = nn.LayerList()
        self.Up_path2 = nn.LayerList()
        self.Skip = nn.LayerList()
        self.Concat = Concat(dim=1)


        n_input = num_input_channels
        for i in range(n_scales):
            self.Down_path.append(Down(n_input, num_channels_down[i], filter_size_down[i],
                                     need_bias=need_bias, act_fun=act_fun,
                                     pad=pad, downsample_mode=downsample_mode))

            if num_channels_skip[i] != 0:
                self.Skip.append(Conv(n_input, num_channels_skip[i], filter_skip_size,
                                      bias=need_bias, act_fun=act_fun, pad=pad))

            if i == n_scales - 1:
                # The deepest
                k = num_channels_down[i]
            else:
                k = num_channels_up[i + 1]

            self.Up_path1.append(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
            self.Up_path2.append(Up(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i],
                                    need_up = need1x1_up, need_bias=need_bias, act_fun=act_fun, pad=pad))

            n_input = num_channels_down[i]

        self.last = nn.LayerList()
        self.last.append(nn.Pad2D(0, mode='reflect'))
        self.last.append(nn.Conv2D(num_channels_up[0], num_output_channels, 1))
        self.need_sigmoid = need_sigmoid

    def forward(self, x):
        skips = []
        for i in range(self.n_scales):
            skips.append(self.Skip[i](x))
            x = self.Down_path[i](x)

        for i in reversed(range(self.n_scales)):
            x = self.Up_path1[i](x)
            x = self.Concat([skips[i], x])
            x = self.Up_path2[i](x)

        x = self.last[0](x)
        x = self.last[1](x)
        if self.need_sigmoid:
            x = nn.functional.sigmoid(x)

        return x