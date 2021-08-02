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


class EmbLoss_v1(nn.Layer):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v1, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = (1.0, 1.0)

    def forward_single(self, emb, instance, kernel, training_mask, bboxes):
        training_mask = (training_mask > 0.5).long()
        kernel = (kernel > 0.5).long()
        instance = instance * training_mask
        instance_kernel = paddle.reshape((instance * kernel),(-1))
        instance = paddle.reshape(instance,(-1))
        emb = paddle.reshape(emb,(self.feature_dim, -1))

        unique_labels, unique_ids = paddle.unique(instance_kernel, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0

        emb_mean = paddle.zeros((self.feature_dim, num_instance), dtype='float32')
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = paddle.mean(emb[:, ind_k], axis=1)

        l_agg = paddle.zeros(num_instance, dtype='float32')
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
            dist = F.relu(dist - self.delta_v) ** 2
            l_agg[i] = paddle.mean(paddle.log(dist + 1.0))
        l_agg = paddle.mean(l_agg[1:])

        if num_instance > 2:
            emb_trans = paddle.transpose(emb_mean, perm=[1, 0])
            emb_interleave = paddle.tile(emb_trans, repeat_times=[num_instance, 1])

            emb_trans = paddle.transpose(emb_mean, perm=[1, 0])
            emb_tile = paddle.tile(emb_trans, repeat_times=[num_instance, 1])
            emb_band = paddle.reshape(emb_tile,(-1, self.feature_dim))
            # print(seg_band)

            mask = (1 - paddle.eye(num_instance, dtype=np.int8))
            mask = paddle.reshape(mask,(-1,1))
            mask = paddle.tile(mask, repeat_times=[1, self.feature_dim])
            mask = paddle.reshape(mask,(num_instance, num_instance, -1))
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = paddle.reshape(mask, (num_instance * num_instance, -1))
            # print(mask)

            dist = emb_interleave - emb_band
            # dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, dim=1)
            dist = paddle.reshape(dist[mask > 0], (-1, self.feature_dim)).norm(p=2, axis=1)

            dist = F.relu(2 * self.delta_d - dist) ** 2
            l_dis = paddle.mean(paddle.log(dist + 1.0))
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = paddle.mean(paddle.log(paddle.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb, instance, kernel, training_mask, bboxes, reduce=True):

        loss_batch = paddle.zeros((emb.size(0)), dtype='float32')

        for i in range(loss_batch.size(0)):
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i], training_mask[i], bboxes[i])

        loss_batch = self.loss_weight * loss_batch

        if reduce:
            loss_batch = paddle.mean(loss_batch)

        return loss_batch
