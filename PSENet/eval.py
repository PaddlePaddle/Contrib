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
import numpy as np
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import ResultFormat, AverageMeter


def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def test(test_loader, model, cfg):
    model.eval()

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_pse_time=AverageMeter(500),
            rec_time=AverageMeter(500),
            total_time=AverageMeter(500)
        )

    for idx, data in enumerate(test_loader):
        img_meta = {}
        img_meta["img_size"] = data[1]
        img_meta["org_img_size"] = data[2]
        data_dict = dict(
            imgs=data[0],
            img_metas=img_meta
        )
        print('Testing %d/%d' % (idx, len(test_loader)))
        sys.stdout.flush()

        # prepare input
        data_dict['imgs'] = data_dict['imgs']
        data_dict.update(dict(
            cfg=cfg
        ))

        # forward
        
        with paddle.no_grad():
            outputs = model(**data_dict)

        if cfg.report_speed:
            report_speed(outputs, speed_meters)

        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        rf.write_result(image_name, outputs)


def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = paddle.io.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        use_shared_memory=True
    )

    device = paddle.get_device()
    paddle.set_device(device)

    # model
    model = build_model(cfg.model)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            sys.stdout.flush()

            checkpoint = paddle.load(args.checkpoint)
            model.set_state_dict(checkpoint)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)

    # test
    test(test_loader, model, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)
