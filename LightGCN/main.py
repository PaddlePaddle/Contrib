
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
import pgl
import paddle
import paddle.nn as nn
from dataloader import BasicDataset, Loader
from model import *
from utils import *
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import config
import time
from pgl.nn import functional as GF
from time import time

from train import *
from eval import * 
from predict import predict
import paddle.distributed as dist
from paddle.distributed import fleet


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
if __name__ == '__main__':

    print(config.config)
    train_dataset = Loader(path=config.dataset)
    name_datasets = config.dataset.split('/')[-1]
    Recmodel = LightGCN(config.config, train_dataset)
    if config.config['multigpu']:
        print('using fleet multigpu training', Recmodel)
        dist.init_parallel_env()
        Recmodel = paddle.DataParallel(Recmodel)
    if config.config['multicpu']:
        fleet.init(is_collective=True)
        optimizer = fleet.distributed_optimizer(optimizer)
        Recmodel = fleet.distributed_model(Recmodel)
        print('using fleet multicpu training', Recmodel)
    Neg_k = 1
    bpr=BPRLoss(Recmodel, config.config)
    f = open (f'logger/train_logger_{name_datasets}.txt','w')
    f_test = open (f'logger/test_logger_{name_datasets}.txt','w')
    

    for epoch in range(config.TRAIN_epochs):
        if epoch %10 == 0:
            cprint("[TEST]")
            preds = predict(train_dataset, Recmodel, epoch, 
            multigpu=config.config['multigpu'], multicpu=config.config['multicpu'])
            result = Test(train_dataset, Recmodel, epoch, multicpu=config.config['multicpu'], multigpu=config.config['multigpu'])
            print(epoch, result, file=f_test, flush=True)
        output_information = BPR_train_original(train_dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=None)
        log_output = f'EPOCH[{epoch+1}/{config.TRAIN_epochs}] {output_information}'
        print(f'EPOCH[{epoch+1}/{config.TRAIN_epochs}] {output_information}')
        print(log_output, file=f, flush=True)
    f.close()
    f_test.close()
    predict_result = predict(train_dataset, Recmodel, epoch, config.config['multicore'])