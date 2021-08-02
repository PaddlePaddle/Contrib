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


class DiceLoss(nn.Layer):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.shape[0]

        input = F.sigmoid(input)

        # input = input.contiguous().view(batch_size, -1)
        # target = target.contiguous().view(batch_size, -1).float()
        # mask = mask.contiguous().view(batch_size, -1).float()

        input = paddle.reshape(input,(batch_size, -1))
        target = paddle.reshape(target,(batch_size, -1))
        target = paddle.cast(target, dtype='float32')
        mask = paddle.reshape(mask,(batch_size, -1))
        mask = paddle.cast(mask, dtype='float32')


        input = input * mask
        target = target * mask

        a = paddle.sum(input * target, axis=1)
        b = paddle.sum(input * input, axis=1) + 0.001
        c = paddle.sum(target * target, axis=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self.loss_weight * loss



        if reduce:
            loss = paddle.mean(loss)


        return loss
