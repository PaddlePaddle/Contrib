import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D,BatchNorm,Linear,Pool2D,Dropout,Sequential
import math
import numpy as np

import rep_flow_layer as rf
#import torch.utils.model_zoo as model_zoo

################
#
# Modified https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# Adds support for B x T x C x H x W video data
#
################


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2D(in_planes, out_planes, filter_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = fluid.layers.relu  #nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2D(inplanes, planes, filter_size=1, bias_attr=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = Conv2D(planes, planes, filter_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = Conv2D(planes, planes * self.expansion, filter_size=1, bias_attr=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = fluid.layers.relu  #nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class repofrep(fluid.dygraph.Layer):
    def __init__(self, name_scope, channels=512):
        super(repofrep, self).__init__(name_scope)
        #self.convframe = Conv3D(num_channels=channels, num_filters=channels, filter_size=[7,1,1], stride=[5,1,1], padding=[1,0,0], bias_attr=False)
        self.rep_flow =  rf.FlowLayer(channels)

        self.rep_flow02 = rf.FlowLayer(channels)


    def forward(self, x):
        #resdual = x
        x = fluid.layers.reshape(x, shape=[-1,16,512,x.shape[2],x.shape[3]])
        #x = fluid.layers.transpose(x, perm=[0,2,1,3,4])
        #x = self.convframe(x)
        x = self.rep_flow(x)
        
        x = self.rep_flow02(x)
        #x = fluid.layers.transpose(x, perm=[0,2,1,3,4])
        x = fluid.layers.reshape(x, shape=[-1, 512,x.shape[3],x.shape[4]])

        return x

class ResNet(fluid.dygraph.Layer):

    def __init__(self, block=BasicBlock, layers=50, inp=3, num_classes=400, input_size=112, dropout=0.5):
        self.inplanes = 64
        self.inp = inp
        super(ResNet, self).__init__()
        self.conv1 = Conv2D(inp, 64, filter_size=7, stride=2, padding=3, 
                               bias_attr=False)
        self.bn1 = BatchNorm(64)
        self.relu = fluid.layers.relu #nn.ReLU(inplace=True)
        self.maxpool = Pool2D(pool_size=3,pool_stride=2,pool_padding=1,pool_type='max')   #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.rep_of_rep = repofrep("flowofflow")
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # probably need to adjust this based on input spatial size
        size = int(math.ceil(input_size/32))
        self.avgpool = Pool2D(pool_size=size,pool_stride=1,pool_padding=0,pool_type='avg')  #nn.AvgPool2d(size, stride=1)
        self.dropout = Dropout(dropout)         #nn.Dropout(p=dropout)
        self.fc = Linear(512 * block.expansion, num_classes)

        #for m in self.sublayers():
         #   if isinstance(m, Conv2D):
          #      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
          #  elif isinstance(m, nn.BatchNorm2d):
           #     nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2D(self.inplanes, planes * block.expansion,
                          filter_size=1, stride=stride, bias_attr=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        # x is BxTxCxHxW
        # spatio-temporal video data
        b,t,c,h,w = x.shape
        # need to view it is B*TxCxHxW for 2D CNN
        # important to keep batch and time axis next to
        # eachother, so a simple view without tranposing is possible
        x = fluid.layers.reshape(x, shape=[b*t,c,h,w])   #x.view(b*t,c,h,w)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.rep_of_rep(x)
         
        x = self.layer3(x)
        x = self.layer4(x)

        #print(x.size())
        x = self.avgpool(x)
        x = fluid.layers.reshape(x, shape=[x.shape[0], -1])  #x.view(x.size(0), -1)
        #x = fluid.layers.reshape(x, shape=[b,t-2,-1])
        x = self.dropout(x)

        # currently making dense, per-frame predictions
        x = self.fc(x)

        # so view as BxTxClass
        x = fluid.layers.reshape(x, shape=[b,t-2,-1])   #x.view(b,t,-1)
        # mean-pool over time
        x = fluid.layers.reduce_mean(x, dim=1)

        x = fluid.layers.softmax(x,axis=1)
        #x = fluid.layers.reshape(x, shape=[-1,self.num_classes])
        return x

        # return BxClass prediction 
        #return x

    def load_state_dict(self, state_dict, strict=True):
        # ignore fc layer
        state_dict = {k:v for k,v in state_dict.items() if 'fc' not in k}
        md = self.state_dict()
        md.update(state_dict)
        # convert to flow representation
        if self.inp != 3:
            for k,v in md.items():
                if k == 'conv1.weight':
                    if isinstance(v, nn.Parameter):
                        v = v.data
                    # change image CNN to 20-channel flow by averaing RGB channels and repeating 20 times
                    v = torch.mean(v, dim=1).unsqueeze(1).repeat(1, self.inp, 1, 1)
                    md[k] = v
        
        super(ResNet, self).load_state_dict(md, strict)


    
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34(pretrained=False, mode='rgb', **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if mode == 'flow':
        model = ResNet(BasicBlock, [3, 4, 6, 3], inp=20, **kwargs)
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(pretrained=False, mode='rgb', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if mode == 'flow':
        model = ResNet(Bottleneck, [3, 4, 6, 3], inp=20, **kwargs)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3]) #, **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model



if __name__ == '__main__':
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        net = resnet50(pretrained=False, mode='rgb')
        #net.to(d)
        i = 1
        for name, parms in net.named_parameters():
            print(name, i)
            i += 1
        img = np.zeros([1, 16, 3, 112, 112]).astype('float32')
        img = fluid.dygraph.to_variable(img)

        print(net(img).numpy().shape)
