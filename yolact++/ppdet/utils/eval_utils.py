# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import time
import cv2

import paddle.fluid as fluid

from .voc_eval import bbox_eval as voc_bbox_eval
from .post_process import mstest_box_post_process, mstest_mask_post_process, box_flip

__all__ = ['parse_fetches', 'eval_run', 'eval_results', 'json_eval_results']

logger = logging.getLogger(__name__)


def parse_fetches(fetches, prog=None, extra_keys=None):
    """
    Parse fetch variable infos from model fetches,
    values for fetch_list and keys for stat
    """
    keys, values = [], []
    cls = []
    for k, v in fetches.items():
        if hasattr(v, 'name'):
            keys.append(k)
            #v.persistable = True
            values.append(v.name)
        else:
            cls.append(v)

    if prog is not None and extra_keys is not None:
        for k in extra_keys:
            try:
                v = fluid.framework._get_var(k, prog)
                keys.append(k)
                values.append(v.name)
            except Exception:
                pass

    return keys, values, cls


def length2lod(length_lod):
    offset_lod = [0]
    for i in length_lod:
        offset_lod.append(offset_lod[-1] + i)
    return [offset_lod]


def get_sub_feed(input, place):
    new_dict = {}
    res_feed = {}
    key_name = ['bbox', 'im_info', 'im_id', 'im_shape', 'bbox_flip']
    for k in key_name:
        if k in input.keys():
            new_dict[k] = input[k]
    for k in input.keys():
        if 'image' in k:
            new_dict[k] = input[k]
    for k, v in new_dict.items():
        data_t = fluid.LoDTensor()
        data_t.set(v[0], place)
        if 'bbox' in k:
            lod = length2lod(v[1][0])
            data_t.set_lod(lod)
        res_feed[k] = data_t
    return res_feed


def clean_res(result, keep_name_list):
    clean_result = {}
    for k in result.keys():
        if k in keep_name_list:
            clean_result[k] = result[k]
    result.clear()
    return clean_result


def eval_run(exe,
             compile_program,
             loader,
             keys,
             values,
             cls,
             cfg=None,
             sub_prog=None,
             sub_keys=None,
             sub_values=None):
    """
    Run evaluation program, return program outputs.
    """
    iter_id = 0
    results = []
    if len(cls) != 0:
        values = []
        for i in range(len(cls)):
            _, accum_map = cls[i].get_map_var()
            cls[i].reset(exe)
            values.append(accum_map)

    images_num = 0
    start_time = time.time()
    has_bbox = 'bbox' in keys

    try:
        loader.start()
        while True:
            outs = exe.run(compile_program,
                           fetch_list=values,
                           return_numpy=False)
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }
            multi_scale_test = getattr(cfg, 'MultiScaleTEST', None)
            mask_multi_scale_test = multi_scale_test and 'Mask' in cfg.architecture

            if multi_scale_test:
                post_res = mstest_box_post_process(res, cfg)
                res.update(post_res)
            if mask_multi_scale_test:
                place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
                sub_feed = get_sub_feed(res, place)
                sub_prog_outs = exe.run(sub_prog,
                                        feed=sub_feed,
                                        fetch_list=sub_values,
                                        return_numpy=False)
                sub_prog_res = {
                    k: (np.array(v), v.recursive_sequence_lengths())
                    for k, v in zip(sub_keys, sub_prog_outs)
                }
                post_res = mstest_mask_post_process(sub_prog_res, cfg)
                res.update(post_res)
            if multi_scale_test:
                res = clean_res(
                    res, ['im_info', 'bbox', 'im_id', 'im_shape', 'mask'])
            if 'mask' in res:
                res['mask'] = mask_encode(res)
            results.append(res)
            if iter_id % 100 == 0:
                logger.info('Test iter {}'.format(iter_id))
            iter_id += 1
            images_num += 1
            # print(res)
            # break
            # images_num += len(res['bbox'][1][0]) if has_bbox else 1
    except (StopIteration, fluid.core.EOFException):
        loader.reset()
    logger.info('Test finish iter {}'.format(iter_id))

    end_time = time.time()
    fps = images_num / (end_time - start_time)
    if has_bbox:
        logger.info('Total number of images: {}, inference time: {} fps.'.
                    format(images_num, fps))
    else:
        logger.info('Total iteration: {}, inference time: {} batch/s.'.format(
            images_num, fps))

    return results

def mask_encode(results, thresh_binarize=0.5):
    import pycocotools.mask as mask_util
    from ppdet.utils.coco_eval import expand_boxes
    # scale = (resolution + 2.0) / resolution
    bboxes = results['bbox'][0]
    masks = results['mask'][0]
    lengths = [len(bboxes)]
    # lengths = results['mask'][1][0]
    im_shapes = results['im_shape'][0]
    segms = []
    if bboxes.shape == (1, 1) or bboxes is None:
        return segms
    if len(bboxes.tolist()) == 0:
        return segms

    s = 0
    # for each sample
    for i in range(len(lengths)):
        num = lengths[i]
        im_shape = im_shapes[i]

        # bbox = bboxes[s:s + num][:, 2:]
        clsid_scores = bboxes[s:s + num][:, 0:2]
        mask = masks[s:s + num]
        s += num

        im_h = int(im_shape[0])
        im_w = int(im_shape[1])
        # expand_bbox = expand_boxes(bbox, scale)
        # expand_bbox = expand_bbox.astype(np.int32)
        # padded_mask = np.zeros(
        #     (resolution + 2, resolution + 2), dtype=np.float32)

        for j in range(num):
            # xmin, ymin, xmax, ymax = expand_bbox[j].tolist()
            clsid, score = clsid_scores[j].tolist()
            clsid = int(clsid)
            # padded_mask[1:-1, 1:-1] = mask[j, clsid, :, :]

            # w = xmax - xmin + 1
            # h = ymax - ymin + 1
            # w = np.maximum(w, 1)
            # h = np.maximum(h, 1)
            
            # resized_mask = cv2.resize(padded_mask, (w, h))
            # resized_mask = np.array(
            #     resized_mask > thresh_binarize, dtype=np.uint8)
            # im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            # x0 = min(max(xmin, 0), im_w)
            # x1 = min(max(xmax + 1, 0), im_w)
            # y0 = min(max(ymin, 0), im_h)
            # y1 = min(max(ymax + 1, 0), im_h)

            # im_mask[y0:y1, x0:x1] = resized_mask[(y0 - ymin):(y1 - ymin), (
            #     x0 - xmin):(x1 - xmin)]
            
            # im_mask = cv2.resize(mask[j, :, :], (im_w, im_h)) > 0.5
            # if np.sum(im_mask) < 300:
            #     thresh_binarize = 0.35
            # elif np.sum(im_mask) > 800:
            #     thresh_binarize = 0.45
            # else:
            #     thresh_binarize = 0.4    
            im_mask = (cv2.resize(mask[j, :, :], (im_w, im_h)) > 0.4).astype(np.uint8)
            segm = mask_util.encode(
                np.array(
                    im_mask[:, :, np.newaxis], order='F'))[0]
            segms.append(segm)
    return segms

def eval_results(results,
                 metric,
                 num_classes,
                 resolution=None,
                 is_bbox_normalized=False,
                 output_directory=None,
                 map_type='11point',
                 dataset=None):
    """Evaluation for evaluation program results"""
    box_ap_stats = []
    if metric == 'COCO':
        from ppdet.utils.coco_eval import proposal_eval, bbox_eval, mask_eval
        anno_file = dataset.get_anno()
        with_background = dataset.with_background
        if 'proposal' in results[0]:
            output = 'proposal.json'
            if output_directory:
                output = os.path.join(output_directory, 'proposal.json')
            proposal_eval(results, anno_file, output)
        if 'bbox' in results[0]:
            output = 'bbox.json'
            if output_directory:
                output = os.path.join(output_directory, 'bbox.json')

            box_ap_stats = bbox_eval(
                results,
                anno_file,
                output,
                with_background,
                is_bbox_normalized=is_bbox_normalized)

        if 'mask' in results[0]:
            output = 'mask.json'
            if output_directory:
                output = os.path.join(output_directory, 'mask.json')
            mask_eval(results, anno_file, output, resolution)
    else:
        if 'accum_map' in results[-1]:
            res = np.mean(results[-1]['accum_map'][0])
            logger.info('mAP: {:.2f}'.format(res * 100.))
            box_ap_stats.append(res * 100.)
        elif 'bbox' in results[0]:
            box_ap = voc_bbox_eval(
                results,
                num_classes,
                is_bbox_normalized=is_bbox_normalized,
                map_type=map_type)
            box_ap_stats.append(box_ap)
    return box_ap_stats


def json_eval_results(metric, json_directory=None, dataset=None):
    """
    cocoapi eval with already exists proposal.json, bbox.json or mask.json
    """
    assert metric == 'COCO'
    from ppdet.utils.coco_eval import cocoapi_eval
    anno_file = dataset.get_anno()
    json_file_list = ['proposal.json', 'bbox.json', 'mask.json']
    if json_directory:
        assert os.path.exists(
            json_directory), "The json directory:{} does not exist".format(
                json_directory)
        for k, v in enumerate(json_file_list):
            json_file_list[k] = os.path.join(str(json_directory), v)

    coco_eval_style = ['proposal', 'bbox', 'segm']
    for i, v_json in enumerate(json_file_list):
        if os.path.exists(v_json):
            cocoapi_eval(v_json, coco_eval_style[i], anno_file=anno_file)
        else:
            logger.info("{} not exists!".format(v_json))
