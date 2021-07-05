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

import argparse
import time
import paddle
import paddle.nn as nn
import numpy as np
import logging
# from data import load_data, read_relation_subsets, gen_rel_subset_feature
from model.model import SIGN, WeightedAggregator
from utils.utils import get_n_params, get_evaluator, train, test

def main(args):
    if args.seed is not None:
        paddle.fluid.Program.random_seed = args.seed
        np.random.seed(args.seed)

    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{args.gpu}"

    # Load dataset
    # data = load_data(device, args)
    # g, labels, num_classes, train_nid, val_nid, test_nid = data
    labels=np.load("./data/lables.npy")
    num_classes=np.load("./data/num_classes.npy")
    train_nid=np.load("./data/train_nid.npy")
    val_nid=np.load("./data/val_nid.npy")
    test_nid=np.load("./data/test_nid.npy")
    evaluator = get_evaluator(args.dataset)

    # Preprocess neighbor-averaged features over sampled relation subgraphs
    rel_subsets =[]
    with paddle.no_grad():
        feats=[]
        for i in range(args.R + 1):
            #数据集请自行在OGB官网下载，并按照官网教程生产训练集，或者在AiStudio上查询data88697
            feature = np.load(f'../data/data88697/feat{i}.npy')
            feats.append(paddle.to_tensor(feature))
        # feats = preprocess_features(g, rel_subsets, args, device)
        print("Done preprocessing")
    # labels = labels.to(device)
    # Release the graph since we are not going to use it later
    g = None

    # Set up logging
    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        level=logging.INFO)
    logging.info(str(args))

    _, num_feats, in_feats = feats[0].shape
    logging.info("new input size: {} {}".format(num_feats,in_feats))

    # Create model
    num_hops =  args.R + 1  # include self feature hop 0
    model = nn.Sequential(
        WeightedAggregator(num_feats, in_feats, num_hops),
        SIGN(in_feats, args.num_hidden, num_classes, num_hops,
             args.ff_layer, args.dropout, args.input_dropout)
    )

    if len(labels.shape) == 1:
        # single label multi-class
        loss_fcn = nn.NLLLoss()
    else:
        # multi-label multi-class
        loss_fcn = nn.KLDivLoss(reduction='batchmean')

    print('!'*100)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),weight_decay=args.weight_decay)
    # optimizer = paddle.optimizer.Adam(parameters=model.parameters())
    # Start training
    best_epoch = 0
    best_val = 0
    f = open('log.txt', 'w+')
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        print(epoch)
        train(model, feats, labels, train_nid, loss_fcn, optimizer, args.batch_size)
        if epoch % args.eval_every == 0:
            with paddle.no_grad():
                train_res, val_res, test_res = test(
                    model, feats, labels, train_nid, val_nid, test_nid, evaluator, args.eval_batch_size)
            end = time.time()
            val_acc = val_res[0]
            log = "Epoch {}, Times(s): {:.4f}".format(epoch, end - start)
            log += ", Accuracy: Train {:.4f}, Val {:.4f}".format(train_res[0], val_res[0])
            log += f", best_acc:{best_val}"
            logging.info(log)
            print(log, file=f, flush=True)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
    f.close()
    logging.info("Best Epoch {}, Val {:.4f}".format(best_epoch, best_val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neighbor-Averaging over Relation Subgraphs")
    parser.add_argument("--num-epochs", type=int, default=1500)
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--R", type=int, default=4,
                        help="number of hops")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="oag")
    parser.add_argument("--data-dir", type=str, default=None, help="path to dataset, only used for OAG")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--eval-batch-size", type=int, default=5000,
                        help="evaluation batch size, -1 for full batch")
    parser.add_argument("--ff-layer", type=int, default=2,
                        help="number of feed-forward layers")
    parser.add_argument("--input-dropout", action="store_true")
    # parser.add_argument("--use-emb", required=True, type=str)
    # parser.add_argument("--use-relation-subsets", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None )
    parser.add_argument("--cpu-preprocess", action="store_true",
                        help="Preprocess on CPU")
    args = parser.parse_args()
    print(args)
    main(args)
