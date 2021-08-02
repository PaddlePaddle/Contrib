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
import paddle.distributed as dist
import numpy as np
import random
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from utils import AverageMeter


dist.get_world_size()
dist.init_parallel_env()

paddle.seed(123456)
np.random.seed(123456)
random.seed(123456)
cnt = 0


def train(train_loader, model, optimizer, epoch, start_iter, cfg):
    model.train()

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()
    accs_rec = AverageMeter()

    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        data_dict = {
            "imgs": data[0],
            "gt_texts": data[1],
            "gt_kernels": data[2],
            "training_masks": data[3]
        }

        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            sys.stdout.flush()
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data_dict.update(dict(cfg=cfg))

        # forward
        outputs = model(**data_dict)

        # detection loss
        loss_text = paddle.mean(outputs['loss_text'])
        losses_text.update(loss_text.numpy())

        loss_kernels = paddle.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.numpy())

        loss = loss_text + loss_kernels

        iou_text = paddle.mean(outputs['iou_text'])
        ious_text.update(iou_text.numpy())
        iou_kernel = paddle.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.numpy())

        losses.update(loss.numpy())
        # backward
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        
        if iter % 20 == 0:
            output_log = '({batch}/{size}) LR: {lr:.6f} | Batch: {bt:.3f}s | Total: {total:.0f}min | ' \
                         'ETA: {eta:.0f}min | Loss: {loss:.3f} | ' \
                         'Loss(text/kernel): {loss_text:.3f}/{loss_kernel:.3f} ' \
                         '| IoU(text/kernel): {iou_text:.3f}/{iou_kernel:.3f} | Acc rec: {acc_rec:.3f}'.format(
                batch=iter + 1,
                size=len(train_loader),
                lr=optimizer.get_lr(),
                bt=batch_time.avg,
                total=batch_time.avg * iter / 60.0,
                eta=batch_time.avg * (len(train_loader) - iter) / 60.0,
                loss_text=losses_text.avg[0],
                loss_kernel=losses_kernels.avg[0],
                loss=losses.avg[0],
                iou_text=ious_text.avg[0],
                iou_kernel=ious_kernel.avg[0],
                acc_rec=accs_rec.avg,
            )
            print(output_log)
            sys.stdout.flush()


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule
    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    optimizer.set_lr(lr)


def save_checkpoint(state, checkpoint_path, cfg):
    param_path = osp.join(checkpoint_path, 'checkpoint_{}_{}.pdparams'.format(state["epoch"],state["iter"]))
    opt_path = osp.join(checkpoint_path, 'checkpoint_{}_{}.pdopt'.format(state["epoch"],state["iter"]))
    paddle.save(state["state_dict"], param_path)
    paddle.save(state["optimizer"], opt_path)


def main(args):
    cfg = Config.fromfile(args.config)
    print(json.dumps(cfg._cfg_dict, indent=4))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('checkpoints', cfg_name)
    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    print('Checkpoint path: %s.' % checkpoint_path)
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.train)
    train_loader = paddle.io.DataLoader(
        data_loader,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        use_shared_memory=True
    )

    # device
    device = paddle.get_device()
    paddle.set_device(device)


    # model
    model = build_model(cfg.model)
    model = paddle.DataParallel(model)

    # Check if model has custom optimizer / loss
    optimizer = None
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
    else:
        if cfg.train_cfg.optimizer == 'SGD':
            optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=cfg.train_cfg.lr,
                                             weight_decay=5e-4, momentum=0.99)

        elif cfg.train_cfg.optimizer == 'Adam':
            optimizer = paddle.optimizer.Adam(parameters = model.parameters(), learning_rate=cfg.train_cfg.lr)

    start_epoch = 0
    start_iter = 0
    if hasattr(cfg.train_cfg, 'pretrain'):
        assert osp.isfile(cfg.train_cfg.pretrain), 'Error: no pretrained weights found!'
        print('Finetuning from pretrained model %s.' % cfg.train_cfg.pretrain)
        checkpoint = paddle.load(cfg.train_cfg.pretrain)
        model.set_state_dict(checkpoint)
    if args.resume:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('checkpoints', cfg_name)
        pdparams_file = checkpoint_path + "/" + args.resume + ".pdparams"
        pdopt_file = checkpoint_path + "/" + args.resume + ".pdopt"
        assert osp.isfile(pdparams_file), 'Error: no checkpoint pdparams file directory found!'
        print('Resuming from checkpoint %s.' % pdparams_file)
        assert osp.isfile(pdopt_file), 'Error: no checkpoint pdopt file directory found!'
        print('Resuming from checkpoint %s.' % pdopt_file)
        start_epoch = int(str(args.resume).split("_")[1])
        start_iter = int(str(args.resume).split("_")[2])
        print("start epoch: ", start_epoch, "   start iter: ", start_iter)
        checkpoint = paddle.load(pdparams_file)
        model.set_state_dict(checkpoint)
        checkpoint = paddle.load(pdopt_file)
        optimizer.set_state_dict(checkpoint)

    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))

        train(train_loader, model, optimizer, epoch, start_iter, cfg)

        state = dict(
            epoch=epoch + 1,
            iter=0,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        if epoch % 1 == 0:
            save_checkpoint(state, checkpoint_path, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    args = parser.parse_args()

    main(args)
