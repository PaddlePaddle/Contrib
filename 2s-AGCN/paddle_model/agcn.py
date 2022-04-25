#encoding=utf8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle as pp
import paddle.nn as nn
import numpy as np
from graph.ntu_rgb_d import Graph


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
        " input size : (N*M, C, T, V)"
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Layer):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = pp.static.create_parameter(shape=A.shape, dtype='float32')
        self.A = pp.to_tensor(A.astype(np.float32))
        self.num_subset = num_subset

        self.conv_a = nn.LayerList()
        self.conv_b = nn.LayerList()
        self.conv_d = nn.LayerList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2D(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2D(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2D(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = pp.transpose(self.conv_a[i](x), perm=[0, 3, 1, 2]).reshape([N, V, self.inter_c * T])
            A2 = self.conv_b[i](x).reshape([N, self.inter_c * T, V])
            A1 = self.soft(pp.matmul(A1, A2) / A1.shape[-1])
            A1 = A1 + A[i]
            A2 = x.reshape([N, C * T, V])
            z = self.conv_d[i](pp.matmul(A2, A1).reshape([N, C, T, V]))
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

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Layer):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=None, in_channels=3):
        super(Model, self).__init__()

        if graph_args is None:
            graph_args = dict()
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
        N, C, T, V, M = x.shape

        x = x.transpose([0, 4, 3, 1, 2]).reshape_([N, M * V * C, T])
        x = self.data_bn(x)
        x = x.reshape_([N, M, V, C, T]).transpose([0, 1, 3, 4, 2]).reshape_([N * M, C, T, V])

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

        # N*M,C,T,V
        c_new = x.shape[1]
        x = x.reshape([N, M, c_new, -1])
        x = x.mean(3).mean(1)

        return self.fc(x)



if __name__ == "__main__":
    # net = unit_tcn(3,3)
    # pp.summary(net, (64, 3, 28, 28))
    graph = Graph()
    # net = unit_gcn(16,16,graph.A)
    # net = TCN_GCN_unit(16,16,graph.A)
    # pp.summary(net, (64, 16, 25, 25))
    net = Model(graph=graph)
    pp.summary(net, (64, 3, 64, 25, 2))











