# encoding=utf8
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
from model.model import *
from utils.utils import *
import paddle
import config
from tqdm import tqdm

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = UniformSample_original_python(dataset)
    users, posItems, negItems = shuffle(S[:, 0], S[:, 1], S[:, 2])
    users = paddle.to_tensor(users, dtype='int64')
    posItems = paddle.to_tensor(posItems, dtype='int64')
    negItems = paddle.to_tensor(negItems, dtype='int64')
    total_batch = len(users) // config.config['bpr_batch_size'] + 1
    aver_loss = 0.
    pbar = tqdm(minibatch(users,
                    posItems,
                    negItems,
                    batch_size=config.config['bpr_batch_size']))
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(pbar):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        pbar.set_description(f'losses: {aver_loss[0]/(batch_i+1)}')
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss[0]:.3f}-{time_info}"

