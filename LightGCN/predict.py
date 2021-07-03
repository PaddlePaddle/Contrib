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
import pgl
import multiprocessing 
import paddle
import paddle.nn as nn
from dataloader import BasicDataset, Loader
from model.model import *
from utils.utils import *
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import config
import time

from pgl.nn import functional as GF
from tqdm import tqdm
from time import time
def predict(dataset, Recmodel, epoch, w=None, multicore=0, multigpu=0, multicpu=0):
    u_batch_size = config.config['test_u_batch_size']
    dataset: BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: LightGCN
    # eval mode with no dropout
    Recmodel.eval()
    max_K = max(config.topks)
    if multicore == 1:
        CORES = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(config.topks)),
               'recall': np.zeros(len(config.topks)),
               'ndcg': np.zeros(len(config.topks))}
    with paddle.no_grad():
        users = list(testDict.keys())
        print(len(users))
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in tqdm(minibatch(users, batch_size=u_batch_size)):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = paddle.to_tensor(batch_users, dtype='int64')
            if multigpu or multicpu:
                rating = Recmodel._layers.getUsersRating(batch_users_gpu)
            else:
                rating = Recmodel.getUsersRating(batch_users_gpu)
           
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = paddle.topk(rating, k=max_K)
            rating = rating.numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        return X