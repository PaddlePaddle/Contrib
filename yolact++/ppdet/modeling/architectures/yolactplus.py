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

from collections import OrderedDict
import copy
import paddle.fluid as fluid

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['Yolactplus']


@register
class Yolactplus(object):
    """
    RetinaNet architecture, see https://arxiv.org/abs/1708.02002

    Args:
        backbone (object): backbone instance
        fpn (object): feature pyramid network instance
        yolact_head (object): `YolactHead` instance
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'fpn', 'yolact_head']

    def __init__(self, backbone, fpn, yolact_head='YolactHead'):
        super(Yolactplus, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.yolact_head = yolact_head
    
    def is_bbox_normalized(self):
        return True
    
    def build(self, feed_vars, mode='train'):
        print('build-----------------------------------------------')
        ['image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd', 'gt_segm']
        if mode == 'train':
            required_fields = ['image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd', 'gt_segm']
        else:
            required_fields = ['im_shape', 'im_info']
        self._input_check(required_fields, feed_vars)
        im = feed_vars['image']
        im_info = feed_vars['im_info']
        if mode == 'train':
            gt_box = feed_vars['gt_bbox']
            gt_class = feed_vars['gt_class']
            gt_segm = feed_vars['gt_segm']
            is_crowd = feed_vars['is_crowd']
            gt_num = feed_vars['gt_num']
        

        mixed_precision_enabled = mixed_precision_global_state() is not None
        # cast inputs to FP16
        if mixed_precision_enabled:
            im = fluid.layers.cast(im, 'float16')

        # backbone
        body_feats = self.backbone(im)

        # cast features back to FP32
        if mixed_precision_enabled:
            body_feats = OrderedDict((k, fluid.layers.cast(v, 'float32'))
                                     for k, v in body_feats.items())

        # FPN
        body_feats, spatial_scale = self.fpn.get_output(body_feats)

        print(type(self.yolact_head))
        
        # retinanet head
        if mode == 'train':
            loss = self.yolact_head.get_loss(body_feats, spatial_scale, im_info,
                                             gt_box, gt_class, gt_segm, is_crowd, gt_num)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.yolact_head.get_prediction(body_feats, spatial_scale,
                                                   im_info)
            return pred

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'test')

    def test(self, feed_vars):
        return self.build(feed_vars, 'test')

    def _input_check(self, require_fields, feed_vars):
        for var in require_fields:
            assert var in feed_vars, \
                "{} has no {} field".format(feed_vars, var)
                
    def _inputs_def(self, image_shape):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image':    {'shape': im_shape,                 'dtype': 'float32',  'lod_level': 0},
            'im_info':  {'shape': [None, 3],                'dtype': 'float32',  'lod_level': 0},
            'im_id':    {'shape': [None, 1],                'dtype': 'int64',    'lod_level': 0},
            'im_shape': {'shape': [None, 3],                'dtype': 'float32',  'lod_level': 0},
            'gt_bbox':  {'shape': [None, 50, 4],            'dtype': 'float32',  'lod_level': 0},
            'gt_segm':  {'shape': [None, 50, 550 , 550],      'dtype': 'float32',  'lod_level': 0},
            'gt_class': {'shape': [None, 50],               'dtype': 'int32',    'lod_level': 0},
            'gt_num':   {'shape': [None, 1],               'dtype': 'int32',    'lod_level': 0},
            'is_crowd': {'shape': [None, 50],               'dtype': 'int32',    'lod_level': 0},
            'is_difficult': {'shape': [None, 50],           'dtype': 'int32',    'lod_level': 0},
        }
        # yapf: enable
        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=[
                'image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd'
            ],  # for-train
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape)
        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in fields])
        loader = fluid.io.DataLoader.from_generator(
            feed_list=list(feed_vars.values()),
            capacity=1,
            use_double_buffer=True,
            iterable=iterable) if use_dataloader else None
        return feed_vars, loader      
        
    # def _inputs_def(self, image_shape, num_max_boxes):
    #     im_shape = [None] + image_shape
    #     # yapf: disable
    #     inputs_def = {
    #         'image':    {'shape': im_shape,                 'dtype': 'float32', 'lod_level': 0},
    #         'im_info':  {'shape': [None, 3],                'dtype': 'float32', 'lod_level': 0},
    #         'im_id':    {'shape': [None, 1],                'dtype': 'int64',   'lod_level': 0},
    #         'im_shape': {'shape': [None, 3],                'dtype': 'float32', 'lod_level': 0},
    #         'gt_bbox':  {'shape': [None, num_max_boxes, 4], 'dtype': 'float32', 'lod_level': 0},
    #         'gt_class': {'shape': [None, num_max_boxes],    'dtype': 'int32',   'lod_level': 0},
    #         'is_crowd': {'shape': [None, num_max_boxes],    'dtype': 'int32',   'lod_level': 0},
    #         'gt_mask':  {'shape': [None, num_max_boxes, 2], 'dtype': 'float32', 'lod_level': 0}, # polygon coordinates
    #         'is_difficult': {'shape': [None, num_max_boxes],'dtype': 'int32',   'lod_level': 0},
    #     }
    #     return inputs_def
        
    # def build_inputs(self,
    #                  image_shape=[3, None, None],
    #                  fields=[
    #                      'image', 'im_info', 'im_id', 'gt_bbox', 'gt_class',
    #                      'is_crowd', 'gt_mask'
    #                  ],
    #                  multi_scale=False,
    #                  num_scales=-1,
    #                  use_flip=None,
    #                  use_dataloader=True,
    #                  iterable=False,
    #                  mask_branch=False):
    #     inputs_def = self._inputs_def(image_shape, 50)
    #     fields = copy.deepcopy(fields)
        
    #     feed_vars = OrderedDict([(key, fluid.data(
    #         name=key,
    #         shape=inputs_def[key]['shape'],
    #         dtype=inputs_def[key]['dtype'],
    #         lod_level=inputs_def[key]['lod_level'])) for key in fields])
    #     use_dataloader = use_dataloader and not mask_branch
    #     loader = fluid.io.DataLoader.from_generator(
    #         feed_list=list(feed_vars.values()),
    #         capacity=64,
    #         use_double_buffer=True,
    #         iterable=iterable) if use_dataloader else None
    #     return feed_vars, loader
    
    
