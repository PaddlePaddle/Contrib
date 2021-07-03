# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import paddle
import numpy as np
from paddle.io import DataLoader, Dataset


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


###############################################################################
# Training and testing for one epoch
###############################################################################
class trainNid(Dataset):
    def __init__(self, train_id):
        self.train_id = train_id

    def __getitem__(self, idx):
        return self.train_id[idx]

    def __len__(self):
        return self.train_id.shape[0]




def train(model, feats, labels, train_nid, loss_fcn, optimizer, batch_size, history=None):
    model.train()
    paddle.set_device('cpu')
    train_nid_list=[]
    for x in train_nid:
        train_nid_temp=[]
        train_nid_temp.append(x)
        train_nid_list.append(train_nid_temp)
    train_nid=np.array(train_nid_list)
    train_nid_data=trainNid(train_nid)

    # print(feats)
    from tqdm import tqdm
    dataloader = DataLoader(train_nid_data, batch_size=batch_size, shuffle=True, drop_last=False)
    pbar = tqdm(dataloader)
    losses = []
    labels = paddle.to_tensor(labels, dtype='int64')
    for batch in pbar:
        batch = batch[0].reshape([1, -1])[0]
        
        batch_feats = [paddle.index_select(x, batch).cuda() for x in feats]
        # batch_feats = [x[batch] for x in feats]
        if history is not None:
            # Train aggregator partially using history
            batch_feats = (batch_feats, [paddle.index_select(x, batch).cuda() for x in history])
        

        loss = loss_fcn(model(batch_feats), paddle.index_select(labels, batch))
        losses.append(loss.cpu().numpy())
        pbar.set_description(f'loss:{np.mean(losses):.3f}')
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()


def test(model, feats, labels, train_nid, val_nid, test_nid, evaluator, batch_size, history=None):
    model.eval()
    num_nodes = labels.shape[0]
    dataset = trainNid(np.arange(num_nodes).reshape(-1, 1))
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, drop_last=False)
    scores = []
    labels = paddle.to_tensor(labels, dtype='int64')
    for batch in dataloader:
        batch = batch[0].reshape([1, -1])[0]
        batch_feats = [paddle.index_select(x, batch) for x in feats]
        if history is not None:
            # Train aggregator partially using history
            batch_feats = (batch_feats, [paddle.index_select(x, batch) for x in history])
        pred = model(batch_feats)
        scores.append(evaluator(pred, paddle.index_select(labels, batch)))
    # For each evaluation metric, concat along node dimension
    metrics = [paddle.concat(s, axis=0) for s in zip(*scores)][0]
    train_res = compute_mean(metrics, train_nid)
    val_res = compute_mean(metrics, val_nid)
    test_res = compute_mean(metrics, test_nid)
    return train_res, val_res, test_res


###############################################################################
# Evaluator for different datasets
###############################################################################

def batched_acc(pred, labels):
    # testing accuracy for single label multi-class prediction
    return (paddle.argmax(pred, axis=1) == labels,)


def get_evaluator(dataset):
    return batched_acc


def compute_mean(metrics, nid):
    num_nodes = len(nid)
    # nid = paddle.to_tensor(nid, dtype='int64')
    metrics = metrics.cpu().numpy()
    # return [paddle.index_select(metrics, nid).mean()]
    return [metrics[nid].mean()]
