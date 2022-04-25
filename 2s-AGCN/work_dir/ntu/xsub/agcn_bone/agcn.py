import numpy as np
import paddle
import paddle.nn as nn
from x2paddle import torch2paddle

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class unit_tcn(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Layer):

    def __init__(self, in_channels, out_channels, A, coff_embedding=4,
                 num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = paddle.create_parameter(
            shape=paddle.to_tensor(A.astype(np.float32)).shape,
            dtype=str(paddle.to_tensor(A.astype(np.float32)).numpy().dtype),
            default_initializer=paddle.nn.initializer.Constant(value=1e-06))
        self.PA.stop_gradient = False
        self.A = paddle.to_tensor(A.astype(np.float32))
        self.num_subset = num_subset
        self.conv_a = nn.LayerList()
        self.conv_b = nn.LayerList()
        self.conv_d = nn.LayerList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2D(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2D(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2D(in_channels, out_channels, 1))
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2D(in_channels, out_channels, 1),
                                      nn.BatchNorm2D(out_channels))
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A
        A = A + self.PA
        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N,
                                                                         V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(paddle.matmul(A1, A2) / A1.size(-1))
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](paddle.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Layer):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size
            =1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Layer):

    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None,
                 graph_args=dict(), in_channels=3):
        super(Model, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = self.graph.A
        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
            N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        return self.fc(x)
