import math
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid import ParamAttr


class Conv1D(fluid.dygraph.Layer):
    def __init__(self,
                 prefix,
                 num_channels=3,
                 num_filters=1,
                 size_k=1,
                 padding=0,
                 groups=1,
                 act=None):
        super(Conv1D, self).__init__()
        fan_in = num_channels * size_k * 1
        k = 1. / math.sqrt(fan_in)
        param_attr = ParamAttr(
            name=prefix + "_w",
            initializer=fluid.initializer.Uniform(
                low=-k, high=k))
        bias_attr = ParamAttr(
            name=prefix + "_b",
            initializer=fluid.initializer.Uniform(
                low=-k, high=k))

        self._conv2d = fluid.dygraph.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=(1, size_k),
            stride=1,
            padding=(0, padding),
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=bias_attr)

    def forward(self, x):
        x = fluid.layers.unsqueeze(input=x, axes=[2])
        x = self._conv2d(x)
        x = fluid.layers.squeeze(input=x, axes=[2])
        return x


class Convlayer(fluid.dygraph.Layer):
    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = Conv2D(1, 64, (1, 3))
        self.conv2 = Conv2D(64, 64, 1)
        self.conv3 = Conv2D(64, 128, 1)
        self.conv4 = Conv2D(128, 256, 1)
        self.conv5 = Conv2D(256, 512, 1)
        self.conv6 = Conv2D(512, 1024, 1)
        self.maxpool = Pool2D(pool_size=(self.point_scales, 1), pool_stride=1)
        self.bn1 = BatchNorm(64, act='relu')
        self.bn2 = BatchNorm(64, act='relu')
        self.bn3 = BatchNorm(128, act='relu')
        self.bn4 = BatchNorm(256, act='relu')
        self.bn5 = BatchNorm(512, act='relu')
        self.bn6 = BatchNorm(1024,  act='relu')

    def forward(self, x):
        x = fluid.layers.unsqueeze(x, 1)
#         x = fluid.layers.relu(self.conv1(x))
#         x = fluid.layers.relu(self.conv2(x))
#         x_128 = fluid.layers.relu(self.conv3(x))
#         x_256 = fluid.layers.relu(self.conv4(x_128))
#         x_512 = fluid.layers.relu(self.conv5(x_256))
#         x_1024 = fluid.layers.relu(self.conv6(x_512))
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x_128 = self.bn3(self.conv3(x))
        x_256 = self.bn4(self.conv4(x_128))
        x_512 = self.bn5(self.conv5(x_256))
        x_1024 = self.bn6(self.conv6(x_512))
        x_128 = fluid.layers.squeeze(input=self.maxpool(x_128), axes=[2])
        x_256 = fluid.layers.squeeze(input=self.maxpool(x_256), axes=[2])
        x_512 = fluid.layers.squeeze(input=self.maxpool(x_512), axes=[2])
        x_1024 = fluid.layers.squeeze(input=self.maxpool(x_1024), axes=[2])
        L = [x_1024, x_512, x_256, x_128]
        x = fluid.layers.concat(L, 1)
        return x


class Latentfeature(fluid.dygraph.Layer):
    def __init__(self, num_scales, each_scales_size, point_scales_list):
        super(Latentfeature, self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = Convlayer(point_scales=self.point_scales_list[0])
        self.Convlayers2 = Convlayer(point_scales=self.point_scales_list[1])
        self.Convlayers3 = Convlayer(point_scales=self.point_scales_list[2])
        self.conv1 = Conv1D(prefix='lf', num_channels=3, num_filters=1, size_k=1, act=None)
        self.bn1 = BatchNorm(1, act='relu')

    def forward(self, x):
        outs = [self.Convlayers1(x[0]), self.Convlayers2(x[1]), self.Convlayers3(x[2])]
        latentfeature = fluid.layers.concat(outs, 2)
        latentfeature = fluid.layers.transpose(latentfeature, perm=[0, 2, 1])
#         latentfeature = fluid.layers.relu(self.conv1(latentfeature))
        latentfeature = self.bn1(self.conv1(latentfeature))
        latentfeature = fluid.layers.squeeze(latentfeature, axes=[1])
        return latentfeature


class PointcloudCls(fluid.dygraph.Layer):
    def __init__(self, num_scales, each_scales_size, point_scales_list, k=40):
        super(PointcloudCls, self).__init__()
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = Linear(1920, 1024)
        self.fc2 = Linear(1024, 512)
        self.fc3 = Linear(512, 256)
        self.fc4 = Linear(256, k)
        # self.dropout = nn.Dropout(p=0.3)
        self.bn1 = BatchNorm(1024, act='relu')
        self.bn2 = BatchNorm(512, act='relu')
        self.bn3 = BatchNorm(256, act='relu')

    def forward(self, x):
        x = self.latentfeature(x)
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        x = self.bn3(self.fc3(x))
        # x = self.bn2(self.dropout(self.fc2(x)))
        # x = self.bn3(self.dropout(self.fc3(x)))
        x = self.fc4(x)
        return fluid.layers.log_softmax(x, axis=1)


class PFNetG(fluid.dygraph.Layer):
    def __init__(self, num_scales, each_scales_size, point_scales_list, crop_point_num):
        super(PFNetG, self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = Linear(input_dim=1920, output_dim=1024, act='relu')
        self.fc2 = Linear(input_dim=1024, output_dim=512, act='relu')
        self.fc3 = Linear(input_dim=512, output_dim=256, act='relu')

        self.fc1_1 = Linear(input_dim=1024, output_dim=128 * 512, act='relu')
        self.fc2_1 = Linear(input_dim=512, output_dim=64 * 128, act='relu')
        self.fc3_1 = Linear(input_dim=256, output_dim=64 * 3)

        self.conv1_1 = Conv1D(prefix='g1_1', num_channels=512, num_filters=512, size_k=1, act='relu')
        self.conv1_2 = Conv1D(prefix='g1_2', num_channels=512, num_filters=256, size_k=1, act='relu')
        self.conv1_3 = Conv1D(prefix='g1_3', num_channels=256, num_filters=int((self.crop_point_num * 3) / 128),
                              size_k=1, act=None)
        self.conv2_1 = Conv1D(prefix='g2_1', num_channels=128, num_filters=6, size_k=1, act=None)

    def forward(self, x):
        x = self.latentfeature(x)
        x_1 = self.fc1(x)  # 1024
        x_2 = self.fc2(x_1)  # 512
        x_3 = self.fc3(x_2)  # 256

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = fluid.layers.reshape(pc1_feat, [-1, 64, 3], inplace=False)

        pc2_feat = self.fc2_1(x_2)
        pc2_feat_reshaped = fluid.layers.reshape(pc2_feat, [-1, 128, 64], inplace=False)
        pc2_xyz = self.conv2_1(pc2_feat_reshaped)  # 6x64 center2

        pc3_feat = self.fc1_1(x_1)
        pc3_feat_reshaped = fluid.layers.reshape(pc3_feat, [-1, 512, 128], inplace=False)
        pc3_feat = self.conv1_1(pc3_feat_reshaped)
        pc3_feat = self.conv1_2(pc3_feat)
        pc3_xyz = self.conv1_3(pc3_feat)  # 12x128 fine

        pc1_xyz_expand = fluid.layers.unsqueeze(pc1_xyz, axes=[2])
        pc2_xyz = fluid.layers.transpose(pc2_xyz, perm=[0, 2, 1])
        pc2_xyz_reshaped1 = fluid.layers.reshape(pc2_xyz, [-1, 64, 2, 3], inplace=False)
        pc2_xyz = fluid.layers.elementwise_add(pc1_xyz_expand, pc2_xyz_reshaped1)
        pc2_xyz_reshaped2 = fluid.layers.reshape(pc2_xyz, [-1, 128, 3], inplace=False)

        pc2_xyz_expand = fluid.layers.unsqueeze(pc2_xyz_reshaped2, axes=[2])
        pc3_xyz = fluid.layers.transpose(pc3_xyz, perm=[0, 2, 1])
        pc3_xyz_reshaped1 = fluid.layers.reshape(pc3_xyz, [-1, 128, int(self.crop_point_num / 128), 3], inplace=False)
        pc3_xyz = fluid.layers.elementwise_add(pc2_xyz_expand, pc3_xyz_reshaped1)
        pc3_xyz_reshaped2 = fluid.layers.reshape(pc3_xyz, [-1, self.crop_point_num, 3], inplace=False)

        return pc1_xyz, pc2_xyz_reshaped2, pc3_xyz_reshaped2  # center1 ,center2 ,fine


# class _netlocalD(nn.Module):
#     def __init__(self, crop_point_num):
#         super(_netlocalD, self).__init__()
#         self.crop_point_num = crop_point_num
#         self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
#         self.conv2 = torch.nn.Conv2d(64, 64, 1)
#         self.conv3 = torch.nn.Conv2d(64, 128, 1)
#         self.conv4 = torch.nn.Conv2d(128, 256, 1)
#         self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.fc1 = nn.Linear(448, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 16)
#         self.fc4 = nn.Linear(16, 1)
#         self.bn_1 = nn.BatchNorm1d(256)
#         self.bn_2 = nn.BatchNorm1d(128)
#         self.bn_3 = nn.BatchNorm1d(16)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x_64 = F.relu(self.bn2(self.conv2(x)))
#         x_128 = F.relu(self.bn3(self.conv3(x_64)))
#         x_256 = F.relu(self.bn4(self.conv4(x_128)))
#         x_64 = torch.squeeze(self.maxpool(x_64))
#         x_128 = torch.squeeze(self.maxpool(x_128))
#         x_256 = torch.squeeze(self.maxpool(x_256))
#         Layers = [x_256, x_128, x_64]
#         x = torch.cat(Layers, 1)
#         x = F.relu(self.bn_1(self.fc1(x)))
#         x = F.relu(self.bn_2(self.fc2(x)))
#         x = F.relu(self.bn_3(self.fc3(x)))
#         x = self.fc4(x)
#         return x
#

# if __name__ == '__main__':

# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("Conv1d") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)
#     elif classname.find("BatchNorm1d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# convlayer_torch = Convlayer(256)
# convlayer_torch.to(device)
# convlayer_torch.apply(weights_init_normal)
# torch.save({'state_dict': convlayer_torch.state_dict()},
#            'Checkpoints/convlayer_torch.pth')

# convlayer_pp = Convlayer_pp(256)


    # input1 = torch.randn(64, 2048, 3)
    # input2 = torch.randn(64, 512, 3)
    # input3 = torch.randn(64, 256, 3)
    # input_ = [input1, input2, input3]
    # netG = _netG(3, 1, [2048, 512, 256], 1024)
    # output = netG(input_)
    # print(output)
