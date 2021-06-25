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

from PIL import Image
import cv2
import paddle.vision.transforms as transforms


def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def predict(path, model, cfg, out_path):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

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

    def get_img(img_path, read_type='pil'):
        try:
            if read_type == 'cv2':
                img = cv2.imread(img_path)
                img = img[:, :, [2, 1, 0]]
            elif read_type == 'pil':
                img = np.array(Image.open(img_path))
        except Exception as e:
            print('Cannot read image: %s.' % img_path)
            raise
        return img

    def scale_aligned_short(img, short_size=736):
        h, w = img.shape[0:2]
        scale = short_size * 1.0 / min(h, w)
        h = int(h * scale + 0.5)
        w = int(w * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        if w % 32 != 0:
            w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))
        return img

    imgs_list = os.listdir(path)
    data_lists = []
    for img_path in imgs_list:
        img = get_img(osp.join(path, img_path))
        img_meta = dict(
            org_img_size=np.array(img.shape[:2])
        )

        img = scale_aligned_short(img, int(cfg.data.test.short_size))
        img_meta.update(dict(
            img_size=np.array(img.shape[:2])
        ))
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        img = img.numpy()
        img = np.expand_dims(img, axis=0)
        img = paddle.to_tensor(img)
        img_meta["img_size"] = paddle.to_tensor(np.expand_dims(img_meta["img_size"], axis=0))
        img_meta["org_img_size"] = paddle.to_tensor(np.expand_dims(img_meta["org_img_size"], axis=0))
        data_lists.append((img, img_meta["img_size"],img_meta["org_img_size"], osp.join(path, img_path)))



    for idx, data in enumerate(data_lists):
        img_meta = {}
        img_meta["img_size"] = data[1]
        img_meta["org_img_size"] = data[2]
        img_path = data[3]
        data_dict = dict(
            imgs=data[0],
            img_metas=img_meta
        )
        print('Testing %d/%d' % (idx, len(data_lists)))
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

        data_type=cfg.data.train.type
        if data_type == "PSENET_IC15" or data_type == "PSENET_IC17":
            save_img = cv2.imread(img_path)
            for bbox in outputs["bboxes"]:
                cv2.rectangle(save_img, (bbox[2], bbox[3]), (bbox[6], bbox[7]), (0, 0, 255), 2)
            save_jpg = img_path.split("/")[-1]
            save_img_path = osp.join(out_path, save_jpg)
            cv2.imwrite(save_img_path, save_img)

        elif data_type == "PSENET_TT":
            save_img = cv2.imread(img_path)
            for bbox in outputs["bboxes"]:
                xs=[]
                ys=[]
                i = 0
                for point in bbox:
                    if i%2==0:
                        xs.append(point)
                    else:
                        ys.append(point)
                    i=i+1
                for idx, x in enumerate(xs):
                    if idx < len(xs)-1:
                        cv2.line(save_img, (x, ys[idx]), (xs[idx+1], ys[idx+1]), (0, 0, 255), 2)
                    else:
                        cv2.line(save_img, (x, ys[idx]), (xs[0], ys[0]), (0, 0, 255), 2)
            save_jpg = img_path.split("/")[-1]
            save_img_path = osp.join(out_path, save_jpg)
            cv2.imwrite(save_img_path, save_img)
                


def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()


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
    predict(args.input, model, cfg, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('input', help='path of images to be predicted')
    parser.add_argument('output', help='path of output', default ="out_imgs")
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)
