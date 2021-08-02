#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from copy import deepcopy, copy


class MAML(paddle.nn.Layer):
    def __init__(self, n_way):
        super(MAML, self).__init__()
        # 定义模型中全部待优化参数
        self.vars = []
        self.vars_bn = []

        # ------------------------第1个conv2d-------------------------
        weight = paddle.static.create_parameter(shape=[64, 1, 3, 3],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        # 第1个BatchNorm
        weight = paddle.static.create_parameter(shape=[64],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------第2个conv2d------------------------
        weight = paddle.static.create_parameter(shape=[64, 64, 3, 3],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])
        # 第2个BatchNorm
        weight = paddle.static.create_parameter(shape=[64],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------第3个conv2d------------------------
        weight = paddle.static.create_parameter(shape=[64, 64, 3, 3],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])
        # 第3个BatchNorm
        weight = paddle.static.create_parameter(shape=[64],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------第4个conv2d------------------------
        weight = paddle.static.create_parameter(shape=[64, 64, 3, 3],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])
        # 第4个BatchNorm
        weight = paddle.static.create_parameter(shape=[64],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),  # 参数可以修改为Xavier
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[64],
                                              dtype='float32',
                                              is_bias=True)  # 初始化为零
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([64], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------全连接层------------------------
        weight = paddle.static.create_parameter(shape=[64, n_way],
                                                dtype='float32',
                                                default_initializer=nn.initializer.XavierNormal(),
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[n_way],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])

    def forward(self, x, params=None, bn_training=True):
        """
        :param x: 输入图片
        :param params:
        :param bn_training: set False to not update
        :return: 输出分类
        """
        if params is None:
            params = self.vars

        weight, bias = params[0], params[1]  # 第1个CONV层
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[2], params[3]  # 第1个BN层
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x)  # 第1个relu
        x = F.max_pool2d(x, kernel_size=2)  # 第1个MAX_POOL层

        weight, bias = params[4], params[5]  # 第2个CONV层
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[6], params[7]  # 第2个BN层
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x)  # 第2个relu
        x = F.max_pool2d(x, kernel_size=2)  # 第2个MAX_POOL层

        weight, bias = params[8], params[9]  # 第3个CONV层
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[10], params[11]  # 第3个BN层
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x)  # 第3个relu
        x = F.max_pool2d(x, kernel_size=2)  # 第3个MAX_POOL层

        weight, bias = params[12], params[13]  # 第4个CONV层
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[14], params[15]  # 第4个BN层
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x)  # 第4个relu
        x = F.max_pool2d(x, kernel_size=2)  # 第4个MAX_POOL层

        x = paddle.reshape(x, [x.shape[0], -1])  ## flatten
        weight, bias = params[-2], params[-1]  # linear
        x = F.linear(x, weight, bias)

        output = x

        return output

    def parameters(self, include_sublayers=True):
        return self.vars


class MetaLearner(nn.Layer):
    def __init__(self, n_way, glob_update_step, glob_update_step_test, glob_meta_lr, glob_base_lr):
        super(MetaLearner, self).__init__()
        self.update_step = glob_update_step  # task-level inner update steps
        self.update_step_test = glob_update_step_test
        self.net = MAML(n_way=n_way)
        self.meta_lr = glob_meta_lr  # 外循环学习率
        self.base_lr = glob_base_lr  # 内循环学习率
        self.meta_optim = paddle.optimizer.Adam(learning_rate=self.meta_lr, parameters=self.net.parameters())
        # self.meta_optim = paddle.optimizer.Momentum(learning_rate=self.meta_lr,
        #                                             parameters=self.net.parameters(),
        #                                             momentum=0.9)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        task_num = x_spt.shape[0]
        query_size = x_qry.shape[1]  # 75 = 15 * 5
        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]

        # 内循环梯度手动更新，外循环梯度使用定义好的更新器更新
        for i in range(task_num):
            # 第0步更新
            y_hat = self.net(x_spt[i], params=None, bn_training=True)  # (setsz, ways)
            loss = F.cross_entropy(y_hat, y_spt[i])
            grad = paddle.grad(loss, self.net.parameters())  # 计算所有loss相对于参数的梯度和

            tuples = zip(grad, self.net.parameters())  # 将梯度和参数一一对应起来
            # fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            # 在query集上测试，计算准确率
            # 这一步使用更新前的数据，loss填入loss_list_qry[0]，预测正确数填入correct_list[0]
            with paddle.no_grad():
                y_hat = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[0] += loss_qry
                pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)  # size = (75)  # axis取-1也行
                correct = paddle.equal(pred_qry, y_qry[i]).numpy().sum().item()
                correct_list[0] += correct
                # 使用更新后的数据在query集上测试。loss填入loss_list_qry[1]，预测正确数填入correct_list[1]
            with paddle.no_grad():
                y_hat = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[1] += loss_qry
                pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)  # size = (75)
                correct = paddle.equal(pred_qry, y_qry[i]).numpy().sum().item()
                correct_list[1] += correct

            # 剩余更新步数
            for k in range(1, self.update_step):
                y_hat = self.net(x_spt[i], params=fast_weights, bn_training=True)
                loss = F.cross_entropy(y_hat, y_spt[i])
                grad = paddle.grad(loss, fast_weights)
                tuples = zip(grad, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

                if k < self.update_step - 1:
                    with paddle.no_grad():
                        y_hat = self.net(x_qry[i], params=fast_weights, bn_training=True)
                        loss_qry = F.cross_entropy(y_hat, y_qry[i])
                        loss_list_qry[k + 1] += loss_qry
                else:  # 对于最后一步update，要记录loss计算的梯度值，便于外循环的梯度传播
                    y_hat = self.net(x_qry[i], params=fast_weights, bn_training=True)
                    loss_qry = F.cross_entropy(y_hat, y_qry[i])
                    loss_list_qry[k + 1] += loss_qry

                with paddle.no_grad():
                    pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)
                    correct = paddle.equal(pred_qry, y_qry[i]).numpy().sum().item()
                    correct_list[k + 1] += correct

        loss_qry = loss_list_qry[-1] / task_num  # 计算最后一次loss的平均值
        self.meta_optim.clear_grad()  # 梯度清零
        loss_qry.backward()
        self.meta_optim.step()

        accs = np.array(correct_list) / (query_size * task_num)  # 计算各更新步数acc的平均值
        loss = np.array(loss_list_qry) / task_num  # 计算各更新步数loss的平均值
        return accs, loss

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        # assert len(x_spt.shape) == 4

        query_size = x_qry.shape[0]
        correct_list = [0 for _ in range(self.update_step_test + 1)]

        new_net = deepcopy(self.net)
        y_hat = new_net(x_spt)
        loss = F.cross_entropy(y_hat, y_spt)
        grad = paddle.grad(loss, new_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

        # 在query集上测试，计算准确率
        # 这一步使用更新前的数据
        with paddle.no_grad():
            y_hat = new_net(x_qry, params=new_net.parameters(), bn_training=True)
            pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)  # size = (75)
            correct = paddle.equal(pred_qry, y_qry).numpy().sum().item()
            correct_list[0] += correct

        # 使用更新后的数据在query集上测试。
        with paddle.no_grad():
            y_hat = new_net(x_qry, params=fast_weights, bn_training=True)
            pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)  # size = (75)
            correct = paddle.equal(pred_qry, y_qry).numpy().sum().item()
            correct_list[1] += correct

        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, params=fast_weights, bn_training=True)
            loss = F.cross_entropy(y_hat, y_spt)
            grad = paddle.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))

            y_hat = new_net(x_qry, fast_weights, bn_training=True)

            with paddle.no_grad():
                pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)
                correct = paddle.equal(pred_qry, y_qry).numpy().sum().item()
                correct_list[k + 1] += correct

        del new_net
        accs = np.array(correct_list) / query_size
        return accs
