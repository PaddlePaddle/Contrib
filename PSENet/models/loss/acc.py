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

EPS = 1e-6

def acc_single(a, b, mask):
    ind = mask == 1
    if paddle.sum(ind) == 0:
        return 0
    correct = (a[ind] == b[ind])#.float()
    correct = paddle.cast(correct, dtype='float32')
    acc = paddle.sum(correct) / correct.size(0)
    return acc

def acc(a, b, mask, reduce=True):
    batch_size = a.size(0)

    a = paddle.reshape(a, (batch_size, -1))
    b = paddle.reshape(b, (batch_size, -1))
    mask = paddle.reshape(mask, (batch_size, -1))
    acc = paddle.zeros((batch_size,), dtype='float32')
    for i in range(batch_size):
        acc[i] = acc_single(a[i], b[i], mask[i])

    if reduce:
        acc = paddle.mean(acc)
    return acc