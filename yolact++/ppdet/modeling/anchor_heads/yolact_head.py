from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


import cv2
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant
from paddle.fluid.regularizer import L2Decay
from ppdet.modeling.ops import (AnchorGenerator, RetinaTargetAssign,
                                RetinaOutputDecoder)

from ppdet.core.workspace import register
import time

def point_form_tensor(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return fluid.layers.concat(input=[boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2], axis=1)
  
__all__ = ['YolactHead']

def create_tmp_var(name, dtype, shape, lod_level=0):
    return fluid.default_main_program().current_block().create_var(
        name=name, dtype=dtype, shape=shape, lod_level=lod_level)
                
def debug_func(x1,x2,x3,x4,x5,x6):
    # print(name)
    print(np.array(x1).shape)
    print(np.array(x2).shape)
    print(np.array(x3).shape)
    print(np.array(x4).shape)
    print(np.array(x5).shape)
    print(np.array(x6).shape)
    
def debug_gt(x1):
    # print(name)
    x = np.array(x1)
    print('debug',x.shape)
    print(np.sum(x))
    print(x)
    # for i in range(50):
    #     if x[0,i] == [0,0,0,0]:
    #         print(i)

from itertools import product
from math import sqrt

def create_anchor(x):
    pred_scales = [[i * 2 ** (j / 3.0) for j in range(3)] for i in [24, 48, 96, 192, 384]]
    anchor_shape = [69, 35, 18, 9, 5]
    # pred_scales = [[i * 2 ** (j / 3.0) for j in range(3)] for i in [384, 192, 96, 48, 24]]
    # anchor_shape = [5, 9, 18, 35, 69]
    aspect_ratios = [[1, 1/2, 2]]
    max_size = 550
    prior_data = []
    for idx in range(len(anchor_shape)):
        conv_w = anchor_shape[idx]
        conv_h = anchor_shape[idx]
        for j, i in product(range(conv_w), range(conv_h)):
            x = (i + 0.5) / conv_w
            y = (j + 0.5) / conv_h   
            for ars in aspect_ratios:
                for scale in pred_scales[idx]:
                    for ar in ars:
                        ar = sqrt(ar)
                        w = scale * ar / max_size
                        h = scale / ar / max_size 
                        
                        prior_data += [x, y, w, h]
    prior_data = np.array(prior_data)
    return np.reshape(prior_data, (-1, 4))

def np_triu(x):
    x = np.array(x)
    return np.triu(x, 1)
        

@register
class YolactHead(object):
    """
    Retina Head

    Args:
        anchor_generator (object): `AnchorGenerator` instance
        target_assign (object): `RetinaTargetAssign` instance
        output_decoder (object): `RetinaOutputDecoder` instance
        num_convs_per_octave (int): Number of convolution layers in each octave
        num_chan (int): Number of octave output channels
        max_level (int): Highest level of FPN output
        min_level (int): Lowest level of FPN output
        prior_prob (float): Used to set the bias init for the class prediction layer
        base_scale (int): Anchors are generated based on this scale
        num_classes (int): Number of classes
    """
    __inject__ = ['anchor_generator']
    __shared__ = ['num_classes']
 
    def __init__(self,
                 anchor_generator=AnchorGenerator().__dict__,
                 num_chan=256,
                 max_level=7,
                 min_level=3,
                 num_classes=81,
                 batch_size=2,
                 num_priors=57744):
        self.anchor_generator = anchor_generator
        self.num_chan = num_chan
        self.max_level = max_level
        self.min_level = min_level
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_priors = num_priors
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)

    def _proto_net(self, body_feats):
        fpn_name_list = list(body_feats.keys())
        fpn_name = fpn_name_list[-1]
        subnet_blob = body_feats[fpn_name]
        filter_list = [256,256,256,'UP',256,32]
        for i in range(len(filter_list)):
            conv_name = 'protonet_conv_n{}_fpn{}'.format(i, 3)
            subnet_blob_in = subnet_blob
            if filter_list[i] == 'UP':
                subnet_blob = fluid.layers.resize_bilinear(
                                subnet_blob_in, scale=2., name='protonet_updown', align_corners=False)    
            else:
                if i == len(filter_list) - 1:
                    subnet_blob = fluid.layers.conv2d(
                        input=subnet_blob_in,
                        num_filters=filter_list[i],
                        filter_size=1,
                        stride=1,
                        padding=0,
                        name=conv_name,
                        param_attr=ParamAttr(
                            name=conv_name + '_w',
                            initializer=Normal(
                                loc=0., scale=0.01)),
                        bias_attr=ParamAttr(
                            name=conv_name + '_b',
                            learning_rate=2.,
                            regularizer=L2Decay(0.)))
                else:
                    subnet_blob = fluid.layers.conv2d(
                        input=subnet_blob_in,
                        num_filters=filter_list[i],
                        filter_size=3,
                        stride=1,
                        padding=1,
                        act='relu',
                        name=conv_name,
                        param_attr=ParamAttr(
                            name=conv_name + '_w',
                            initializer=Normal(
                                loc=0., scale=0.01)),
                        bias_attr=ParamAttr(
                            name=conv_name + '_b',
                            learning_rate=2.,
                            regularizer=L2Decay(0.)))
        return subnet_blob
        
    def _semantic_seg_conv(self, body_feats):
        fpn_name_list = list(body_feats.keys())
        fpn_name = fpn_name_list[-1]
        subnet_blob = body_feats[fpn_name]
        conv_name = 'semantic_conv_fpn'
        subnet_blob_in = subnet_blob
        subnet_blob = fluid.layers.conv2d(
            input=subnet_blob_in,
            num_filters=80,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name=conv_name,
            param_attr=ParamAttr(
                name=conv_name + '_w',
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=conv_name + '_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        return subnet_blob

    def _prediction_layers(self, body_feats):
        fpn_name_list = list(body_feats.keys())
        bbox_list = []
        conf_list = []
        mask_list = []
        for lvl in range(self.min_level, self.max_level + 1):
            fpn_name = fpn_name_list[self.max_level - lvl]
            subnet_blob = body_feats[fpn_name]
            conv_name = 'upfeature_conv_n_fpn{}'.format(lvl)
            conv_share_name = 'upfeature_conv_n_fpn_share'
            subnet_blob_in = fluid.layers.conv2d(
                    input=subnet_blob,
                    num_filters=self.num_chan,
                    filter_size=3,
                    stride=1,
                    padding=1,
                    act='relu',
                    name=conv_name,
                    param_attr=ParamAttr(
                        name=conv_share_name + '_w',
                        initializer=Normal(
                            loc=0., scale=0.01)),
                    bias_attr=ParamAttr(
                        name=conv_share_name + '_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.)))
            # bias_init = float(-np.log((1 - self.prior_prob) / self.prior_prob))
            #bbox_layer
            bbox_name = 'protonet_bbox_pred_fpn{}'.format(lvl)
            bbox_share_name = 'protonet_bbox_pred_fpn'
            out_bbox = fluid.layers.conv2d(
                input=subnet_blob_in,
                num_filters=36,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                name=bbox_name,
                param_attr=ParamAttr(
                    name=bbox_share_name + '_w',
                    initializer=Normal(
                        loc=0., scale=0.01)),
                bias_attr=ParamAttr(
                    name=bbox_share_name + '_b',
                    # initializer=Constant(value=bias_init),
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))
                    
            out_bbox_transpose = fluid.layers.transpose(
                out_bbox, perm=[0, 2, 3, 1])
            out_bbox_reshape = fluid.layers.reshape(
                out_bbox_transpose, shape=(0, -1, 4))
            out_bbox_reshape = fluid.layers.cast(
                out_bbox_reshape, 'float32')
            bbox_list.append(out_bbox_reshape)
            
            
            
            #conf_layer
            conf_name = 'protonet_conf_pred_fpn{}'.format(lvl)
            conf_share_name = 'protonet_conf_pred_fpn'
            out_conf = fluid.layers.conv2d(
                input=subnet_blob_in,
                num_filters=729,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                name=conf_name,
                param_attr=ParamAttr(
                    name=conf_share_name + '_w',
                    initializer=Normal(
                        loc=0., scale=0.01)),
                bias_attr=ParamAttr(
                    name=conf_share_name + '_b',
                    # initializer=Constant(value=bias_init),
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))
            out_conf_transpose = fluid.layers.transpose(
                out_conf, perm=[0, 2, 3, 1])
            out_conf_reshape = fluid.layers.reshape(
                out_conf_transpose, shape=(0, -1, 81))  
             
            # fluid.layers.py_func(func=debug_gt, x=out_conf_reshape ,out=None)   
            conf_list.append(out_conf_reshape)
            
            #mask_layer
            mask_name = 'protonet_mask_pred_fpn{}'.format(lvl)
            mask_share_name = 'protonet_mask_pred_fpn'
            out_mask = fluid.layers.conv2d(
                input=subnet_blob_in,
                num_filters=288,
                filter_size=3,
                stride=1,
                padding=1,
                act='tanh',
                name=mask_name,
                param_attr=ParamAttr(
                    name=mask_share_name + '_w',
                    initializer=Normal(
                        loc=0., scale=0.01)),
                bias_attr=ParamAttr(
                    name=mask_share_name + '_b',
                    # initializer=Constant(value=bias_init),
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))
            out_mask_transpose = fluid.layers.transpose(
                out_mask, perm=[0, 2, 3, 1])
            out_mask_reshape = fluid.layers.reshape(
                out_mask_transpose, shape=(0, -1, 32)) 
                
            mask_list.append(out_mask_reshape)
            
        return bbox_list, conf_list, mask_list
        
    def _get_output(self, body_feats, spatial_scale):
        """
        Get class, bounding box predictions and anchor boxes of all level FPN level.

        Args:
            fpn_dict(dict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.

        Returns:
            cls_pred_input(list): Class prediction of all input fpn levels.
            bbox_pred_input(list): Bounding box prediction of all input fpn
                levels.
            anchor_input(list): Anchors of all input fpn levels with shape of.
            anchor_var_input(list): Anchor variance of all input fpn levels with
                shape.
        """
        assert len(body_feats) == self.max_level - self.min_level + 1
        #proto_net
        proto_pred_list = self._proto_net(body_feats)
        proto_pred_reshape_list = fluid.layers.transpose(
                proto_pred_list, perm=[0, 2, 3, 1])
        
        #prediction_layers
        bbox_pred_reshape_list, conf_pred_reshape_list, mask_pred_reshape_list = self._prediction_layers(body_feats)
                                                    
        #generate segm
        segm_pred_list = self._semantic_seg_conv(body_feats)
            
        norm_anchor = create_tmp_var('norm_anchor', 'float32', [self.num_priors, 4])
        fluid.layers.py_func(func=create_anchor, x=norm_anchor ,out=norm_anchor)
        # fluid.layers.py_func(func=debug_gt, x=norm_anchor ,out=None)
        
        output = {}
        output['loc'] = bbox_pred_reshape_list
        output['conf'] = conf_pred_reshape_list
        output['mask'] = mask_pred_reshape_list
        output['anchor'] = norm_anchor
        output['proto'] = proto_pred_reshape_list
        output['segm'] = segm_pred_list
        return output
        
        
    def decode(self, loc, priors):
        variances = [0.1, 0.2]
        
        boxes = fluid.layers.concat([
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * fluid.layers.exp(loc[:, 2:] * variances[1])], 1)
        # boxes[:, :2] -= boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]

        x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
        x2y2 = boxes[:, 2:] + x1y1
        
        boxes = fluid.layers.concat([x1y1, x2y2], -1)
        
        return boxes  

    def fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        
        # fluid.layers.py_func(func=debug_shape, x=scores ,out=None)
        scores, idx = fluid.layers.argsort(scores, axis=1, descending=True)
        
        idx = idx[:, :top_k]
        scores = scores[:, :top_k]
        
        idx_shape = fluid.layers.shape(idx)
        num_classes = idx_shape[0] 
        num_dets = idx_shape[1] 
        
        idx = fluid.layers.reshape(idx, [-1])
        
        boxes = fluid.layers.reshape(fluid.layers.gather(boxes, idx), (num_classes, num_dets, 4))
        masks = fluid.layers.reshape(fluid.layers.gather(masks, idx), (num_classes, num_dets, -1))
        
        iou = jaccard_tensor_3D(boxes, boxes)
        fluid.layers.py_func(func=np_triu, x=iou ,out=iou)
        iou_max = fluid.layers.reduce_max(iou, 1)
        
        keep = (iou_max <= iou_threshold)
        
        if second_threshold:
            conf_thresh = 0.05
            keep *= fluid.layers.cast(scores > conf_thresh, 'float32')
        
        classes = fluid.layers.expand(fluid.layers.unsqueeze(fluid.layers.range(0, num_classes, 1, 'int32'), 1), (1, num_dets))
        out = fluid.layers.where(keep)
        
        classes = fluid.layers.gather_nd(classes, out)
        
        boxes = fluid.layers.gather_nd(boxes, out)
        masks = fluid.layers.gather_nd(masks, out)
        scores = fluid.layers.gather_nd(scores, out) 
        
        scores, idx = fluid.layers.argsort(scores, axis=0, descending=True)
        max_num_detections = 100
        idx = idx[:max_num_detections]
        scores = scores[:max_num_detections]
        
        classes = fluid.layers.gather(classes, idx)
        boxes = fluid.layers.gather(boxes, idx)
        masks = fluid.layers.gather(masks, idx)
        
        return boxes, masks, classes, scores
        
    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, use_fastnms=False):
        
        if use_fastnms:
            cur_scores = conf_preds[batch_idx, 1:, :]
            conf_scores = fluid.layers.reduce_max(cur_scores, dim=0)
            conf_thresh = 0.05
            nms_thresh = 0.5
            keep = (conf_scores > conf_thresh)
            
            out = fluid.layers.where(keep)
            # fluid.layers.py_func(func=debug_gt, x=out ,out=None) 
            
            one = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
            pred = fluid.layers.less_than(fluid.layers.shape(out)[0], one) 
            out1 = fluid.layers.reshape(fluid.layers.range(0, fluid.layers.shape(decoded_boxes)[0], 1, 'int32'), (-1,1))
            # fluid.layers.py_func(func=debug_shape, x=out1 ,out=None) 
            #静态图判断不太会，只能这么代替了
            out_res = fluid.layers.cond(pred, lambda: out1, lambda: out)
            
            
            scores = fluid.layers.transpose(cur_scores, [1,0])
            scores = fluid.layers.gather(scores, out_res)
            scores = fluid.layers.transpose(scores, [1,0])
            boxes = fluid.layers.gather(decoded_boxes, out_res)
            masks = fluid.layers.gather(mask_data[batch_idx], out_res)
            nms_top_k = 200
            boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, nms_thresh, nms_top_k)
            # fluid.layers.py_func(func=debug_gt, x=boxes ,out=None)
            return boxes, masks, classes, scores
        else:
            boxes = decoded_boxes
            scores = conf_preds
            masks = mask_data[batch_idx]
            boxes = fluid.layers.unsqueeze(boxes, [0])
            out, index = fluid.contrib.layers.multiclass_nms2(boxes, 
                        scores, 
                        score_threshold=0.05, 
                        nms_top_k=1000, 
                        keep_top_k=100, 
                        nms_threshold=0.5, 
                        normalized=True, 
                        background_label=0,
                        return_index=True)
            
            return out, index, masks, decoded_boxes
        
        
        # if scores.size(1) == 0:
        #     return None     
    
    def get_prediction(self, body_feats, spatial_scale, im_info):
        """
        Get prediction bounding box in test stage.

        Args:
            fpn_dict(dict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.
            im_info (Variable): A 2-D LoDTensor with shape [B, 3]. B is the
                number of input images, each element consists of im_height,
                im_width, im_scale.

        Returns:
            pred_result(Variable): Prediction result with shape [N, 6]. Each
                row has 6 values: [label, confidence, xmin, ymin, xmax, ymax].
                N is the total number of prediction.
        """
        output = self._get_output(body_feats, spatial_scale)
        bbox_pred_reshape_list = output['loc']
        conf_pred_reshape_list = output['conf']
        mask_pred_reshape_list = output['mask']
        anchor_reshape_list = output['anchor']
        proto_data = output['proto']
        segm_pred_reshape_list = output['segm']

        loc_data = fluid.layers.concat(bbox_pred_reshape_list, axis=1)
        conf_data = fluid.layers.concat(conf_pred_reshape_list, axis=1)
        conf_data = fluid.layers.softmax(conf_data, use_cudnn=True)
        mask_data = fluid.layers.concat(mask_pred_reshape_list, axis=1)
        prior_data = anchor_reshape_list
        prior_data = fluid.layers.cast(prior_data, 'float32')

        # fluid.layers.py_func(func=debug_sum, x=loc_data ,out=None)
        # fluid.layers.py_func(func=debug_sum, x=conf_data ,out=None)
        # fluid.layers.py_func(func=debug_sum, x=mask_data ,out=None)
        # fluid.layers.py_func(func=debug_sum, x=proto_data ,out=None)
        # fluid.layers.py_func(func=debug_sum, x=segm_pred_reshape_list ,out=None)
        
        batch_size = fluid.layers.shape(loc_data)[0]
        num_priors = fluid.layers.shape(prior_data)[0]
        
        conf_preds = fluid.layers.transpose(fluid.layers.reshape(conf_data, (batch_size, num_priors, 81)), [0, 2, 1])
        
        decoded_boxes = self.decode(loc_data[0], prior_data)
        
        use_fastnms = False
        if use_fastnms:
            boxes, masks, classes, scores = self.detect(0, conf_preds, decoded_boxes, mask_data, use_fastnms=True)
            
            
            masks = fluid.layers.matmul(proto_data[0], masks, transpose_x=False, transpose_y=True)
            masks = fluid.layers.sigmoid(masks)
            
            masks = crop_tensor(masks, boxes)
            masks = fluid.layers.transpose(masks, [2, 0, 1])
            
            maskiou_p = self.mask_iou_net([8, 16, 32, 64, 128, 80], fluid.layers.unsqueeze(masks, 1))
            
            idx_s = fluid.layers.range(0, fluid.layers.shape(classes)[0], 1, 'int32')
            idx_s = fluid.layers.reshape(idx_s, (-1,1))
            classes = fluid.layers.reshape(classes, (-1,1))

            transform_idx_t = fluid.layers.concat([idx_s, classes], -1)
            maskiou_p = fluid.layers.gather_nd(maskiou_p, transform_idx_t) 
            maskiou_p = fluid.layers.reshape(maskiou_p, shape=(-1,1))
            bbox_scores = fluid.layers.reshape(scores, [-1,1])
            mask_scores = bbox_scores * maskiou_p
            
            mask_scores, idx = fluid.layers.argsort(mask_scores, axis=0, descending=True)
            classes = fluid.layers.gather(classes, idx)
            classes = fluid.layers.cast(classes, 'float32')
            masks = fluid.layers.cast(masks, 'float32')  
            masks = fluid.layers.gather(masks, idx)
            boxes = fluid.layers.gather(boxes, idx)
            box_pred = fluid.layers.concat([classes+1, mask_scores, boxes], -1)
        else:
            box_pred, index, masks, boxes = self.detect(0, conf_preds, decoded_boxes, mask_data, use_fastnms=False)
            scores = box_pred[:, 1]
            boxes = box_pred[:, 2:6]
            classes = box_pred[:, 0]

            masks = fluid.layers.gather(masks, index)
            masks = fluid.layers.matmul(proto_data[0], masks, transpose_x=False, transpose_y=True)
            masks = fluid.layers.sigmoid(masks)
            masks = crop_tensor(masks, boxes)
            masks = fluid.layers.transpose(masks, [2, 0, 1])

            maskiou_p = self.mask_iou_net([8, 16, 32, 64, 128, 80], fluid.layers.unsqueeze(masks, 1))

            idx_s = fluid.layers.range(0, fluid.layers.shape(classes)[0], 1, 'int32')
            idx_s = fluid.layers.reshape(idx_s, (-1,1))
            classes = fluid.layers.reshape(classes, (-1,1))

            # fluid.layers.py_func(func=debug_shape, x=idx_s ,out=None)
            # fluid.layers.py_func(func=debug_shape, x=classes ,out=None)

            transform_idx_t = fluid.layers.concat([idx_s, fluid.layers.cast(classes, 'int32')], -1)
            maskiou_p = fluid.layers.gather_nd(maskiou_p, transform_idx_t) 
            maskiou_p = fluid.layers.reshape(maskiou_p, shape=(-1,1))
            bbox_scores = fluid.layers.reshape(scores, [-1,1])
            mask_scores = bbox_scores * maskiou_p
            
            mask_scores, idx = fluid.layers.argsort(mask_scores, axis=0, descending=True)
            classes = fluid.layers.gather(classes, idx)
            classes = fluid.layers.cast(classes, 'float32')
            masks = fluid.layers.cast(masks, 'float32')  
            masks = fluid.layers.gather(masks, idx)
            boxes = fluid.layers.gather(boxes, idx)
            box_pred = fluid.layers.concat([classes, mask_scores, boxes], -1)           

        # mask_pred = fluid.layers.concat([classes, bbox_scores, masks], -1)
        # fluid.layers.py_func(func=debug_shape, x=box_pred ,out=None)
        # return {'bbox': box_pred, 'mask': mask_pred}
        
        return {'bbox': box_pred, 'mask': masks}

    def get_loss(self, body_feats, spatial_scale, im_info, gt_box, gt_class, gt_mask, is_crowd, gt_num):
        """
        Calculate the loss of retinanet.
        Args:
            fpn_dict(dict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.
            im_info(Variable): A 2-D LoDTensor with shape [B, 3]. B is the
                number of input images, each element consists of im_height,
                im_width, im_scale.
            gt_box(Variable): The ground-truth bounding boxes with shape [M, 4].
                M is the number of groundtruth.
            gt_label(Variable): The ground-truth labels with shape [M, 1].
                M is the number of groundtruth.
            is_crowd(Variable): Indicates groud-truth is crowd or not with
                shape [M, 1]. M is the number of groundtruth.

        Returns:
            Type: dict
                loss_cls(Variable): focal loss.
                loss_bbox(Variable): smooth l1 loss.
        """
        output = self._get_output(body_feats, spatial_scale)
        bbox_pred_reshape_list = output['loc']
        conf_pred_reshape_list = output['conf']
        mask_pred_reshape_list = output['mask']
        anchor_reshape_list = output['anchor']
        proto_data = output['proto']
        segm_pred_reshape_list = output['segm']

        # fluid.layers.py_func(func=debug_shape, x=segm_pred_reshape_list ,out=None)

        
        loc_data = fluid.layers.concat(bbox_pred_reshape_list, axis=1)
        conf_data = fluid.layers.concat(conf_pred_reshape_list, axis=1)
        mask_data = fluid.layers.concat(mask_pred_reshape_list, axis=1)
        priors = anchor_reshape_list
        priors = fluid.layers.cast(priors, 'float32')
        priors.stop_gradient = True

        # fluid.layers.py_func(func=debug_func, x=[loc_data, conf_data, mask_data, priors, proto_data, segm_pred_reshape_list] ,out=None)
        # fluid.layers.py_func(func=debug_sum, x=loc_data ,out=None)
        # fluid.layers.py_func(func=debug_sum, x=conf_data ,out=None)
        # fluid.layers.py_func(func=debug_sum, x=mask_data ,out=None)
        # fluid.layers.py_func(func=debug_sum, x=proto_data ,out=None)
        # fluid.layers.py_func(func=debug_sum, x=segm_pred_reshape_list ,out=None)
        # point_form_tensor(priors)
        
        # batch_size = self.batch_size
        # num_priors = self.num_priors
        # num_classes = self.num_classes
        
        batch_size = self.batch_size
        num_priors = self.num_priors
        num_classes = self.num_classes
        
        loc_t, gt_box_t, conf_t, idx_t, pos = get_target_tensor(gt_box, priors, gt_class, is_crowd, gt_num, loc_data, batch_size, num_priors)
        labels = gt_class
        
        # loc_t = create_tmp_var('loc_t','float32',[batch_size, num_priors, 4])
        # gt_box_t = create_tmp_var('gt_box_t','float32',[batch_size, num_priors, 4])
        # conf_t = create_tmp_var('conf_t','int32',[batch_size, num_priors])
        # idx_t = create_tmp_var('idx_t','int32',[batch_size, num_priors])
        
        # pos = create_tmp_var('pos','bool',[batch_size, num_priors])
        
        # labels = create_tmp_var('labels','int32',[batch_size, 50])
            
        
        # fluid.layers.py_func(func=get_target, x=[gt_box, priors, gt_class, is_crowd, gt_num, loc_data] ,out=[loc_t, gt_box_t, conf_t, idx_t, pos, labels])
    
        
        # loc_t.stop_gradient=True
        # conf_t.stop_gradient=True
        # idx_t.stop_gradient=True
        # labels.stop_gradient=True
        # gt_box_t.stop_gradient=True
        # pos.stop_gradient=True
        # num_pos.stop_gradient=True
        # pos_idx.stop_gradient=True
        
        
        losses = {}
        
        
        out = fluid.layers.where(pos)
        out.stop_gradient=True
        # fluid.layers.py_func(func=debug_gt, x=out ,out=None)
        loc_pt = fluid.layers.gather_nd(loc_data, out)
        loc_tt = fluid.layers.gather_nd(loc_t, out)
    
        # fluid.layers.py_func(func=debug_gt, x=loc_pt ,out=None)
        # fluid.layers.py_func(func=debug_gt, x=loc_tt ,out=None)
        loc_tt.stop_gradient = True
        loss_bbox = fluid.layers.smooth_l1(
            x=fluid.layers.cast(loc_pt,'float32'),
            y=fluid.layers.cast(loc_tt,'float32'))
        losses['B'] = fluid.layers.reduce_sum(loss_bbox) * 1.5
        # fluid.layers.py_func(func=debug_gt, x=losses['B'] ,out=None)
        
        losses['M'], maskiou_targets = self.lincomb_mask_loss(pos, idx_t, mask_data, proto_data, gt_mask, gt_box_t, labels, gt_num, batch_size, num_priors)
        # losses.update(loss)
        # fluid.layers.py_func(func=debug_gt, x=losses['M'] ,out=None)   
        
        losses['C'] = self.ohem_conf_loss(conf_data, conf_t, pos, batch_size, num_priors)
        # fluid.layers.py_func(func=debug_gt, x=losses['C'] ,out=None)
        
        losses['S'] = self.semantic_segmentation_loss(segm_pred_reshape_list, gt_mask, labels, batch_size, gt_num)
        # fluid.layers.py_func(func=debug_gt, x=losses['S'] ,out=None)
        
        # # # aa = self.mask_iou_loss([8, 16, 32, 64, 128, 80], maskiou_targets)
        losses['I'] = self.mask_iou_loss([8, 16, 32, 64, 128, 80], maskiou_targets)
        
        total_num_pos = fluid.layers.reduce_sum(fluid.layers.cast(pos, 'int32'))
        # fluid.layers.py_func(func=debug_gt, x=total_num_pos ,out=None)
        
        for k in losses:
            if k not in ('P', 'E', 'S'):
                losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size        
        # fluid.layers.py_func(func=debug_gt, x=losses['B'] ,out=None)
        # print('loss------------------------------------------------------')
        return losses
        

    def mask_iou_net(self, net, maskiou_net_input):
        stride_list = [2,2,2,2,2,1]
        kernel_list = [3,3,3,3,3,1]        
        subnet_blob_in = maskiou_net_input
        # fluid.layers.py_func(func=debug_gt, x=maskiou_t ,out=None)
        for i in range(len(net)):
            # fluid.layers.py_func(func=debug_shape, x=subnet_blob_in ,out=None)
            conv_name = 'maskiou_conv_n{}'.format(i)
            subnet_blob = fluid.layers.conv2d(
                input=subnet_blob_in,
                num_filters=net[i],
                filter_size=kernel_list[i],
                stride=stride_list[i],
                padding="SAME",
                act='relu',
                name=conv_name,
                param_attr=ParamAttr(
                    name=conv_name + '_w',
                    initializer=Normal(
                        loc=0., scale=0.01)),
                bias_attr=ParamAttr(
                    name=conv_name + '_b',
                    learning_rate=2.,
                    regularizer=L2Decay(0.))) 
            subnet_blob_in = subnet_blob
        maskiou_p = subnet_blob_in
        maskiou_p = fluid.layers.pool2d(maskiou_p, global_pooling=True)
        maskiou_p = fluid.layers.squeeze(maskiou_p, axes=[2,3])
        return maskiou_p
        
    def mask_iou_loss(self, net, maskiou_targets): 
        maskiou_net_input, maskiou_t, label_t = maskiou_targets
        maskiou_p = self.mask_iou_net(net, maskiou_net_input)
        
        idx_s = fluid.layers.range(0, fluid.layers.shape(label_t)[0], 1, 'int32')
        idx_s = fluid.layers.reshape(idx_s, (-1,1))
        label_t = fluid.layers.reshape(label_t, (-1,1))
        transform_idx_t = fluid.layers.concat([idx_s, label_t], -1)
        transform_idx_t.stop_gradient=True
        maskiou_p = fluid.layers.gather_nd(maskiou_p, transform_idx_t)
        maskiou_p = fluid.layers.reshape(maskiou_p, shape=(-1,1))
        maskiou_t = fluid.layers.reshape(maskiou_t, shape=(-1,1))
        maskiou_t.stop_gradient=True
        loss = fluid.layers.smooth_l1(maskiou_p,maskiou_t)
        loss = fluid.layers.reduce_sum(loss) * 25
        return loss
    
    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, batch_size, gt_num):
        num_classes = 80
        mask_h = 69
        mask_w = 69
        loss_s = 0
        
        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]
            cur_num = gt_num[idx]
            
            # segment_t = create_tmp_var('segment_t','float32',[80, 72, 72])
            
            # fluid.layers.py_func(func=get_segment_t, x=[mask_t[idx], cur_class_t, cur_num] ,out=[segment_t])
            
            segment_t = get_segment_t_tensor(mask_t[idx], cur_class_t, cur_num, mask_w, mask_h)
            segment_t.stop_gradient = True
            # fluid.layers.py_func(func=debug_sum, x=cur_segment ,out=None)
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=fluid.layers.reshape(cur_segment, shape=(80,-1)),
                    label=fluid.layers.reshape(segment_t, shape=(80,-1)))
            
            loss_s += fluid.layers.reduce_sum(loss)
            # fluid.layers.py_func(func=debug_sum, x=fluid.layers.reduce_sum(loss) ,out=None)
            
        loss_s = loss_s / mask_h / mask_w
        return loss_s
        
    def ohem_conf_loss(self, conf_data, conf_t, pos, num, num_priors):
         
         
        # fluid.layers.py_func(func=debug_max, x=conf_data ,out=None)
        
        # posneg_idx = create_tmp_var('posneg_idx','bool',[num * num_priors * 81, 1])
        # t = create_tmp_var('t','bool',[num, num_priors])
        # posneg = create_tmp_var('posneg','bool',[num*num_priors, 1])
        # fluid.layers.py_func(func=get_posneg, x=[conf_data, conf_t, pos], out=[t, posneg, posneg_idx])
        
        batch_conf = fluid.layers.reshape(conf_data, shape=(-1, 81))
        x_max = fluid.layers.reduce_max(batch_conf)
        loss_c = fluid.layers.log(fluid.layers.reduce_sum(fluid.layers.exp(batch_conf - x_max), dim=1)) + x_max
        loss_c = loss_c - batch_conf[:, 0]
        
        batch_size = fluid.layers.shape(conf_data)[0]
        
        loss_c = fluid.layers.reshape(loss_c, (batch_size, -1))
        pos = fluid.layers.cast(pos, 'int32')

        loss_c *= -(pos - 1) 
        loss_c *= -(fluid.layers.cast(conf_t < 0, 'int32') - 1)
        
        _, loss_idx = fluid.layers.argsort(loss_c, 1, descending=True)
        _, idx_rank = fluid.layers.argsort(loss_idx, 1)
        idx_rank = fluid.layers.cast(idx_rank, 'int32')
        
        
        num_pos = fluid.layers.reduce_sum(pos, dim=1, keep_dim=True)
        # fluid.layers.py_func(func=debug_gt, x=num_pos ,out=None)
        negpos_ratio = 3
        num_neg = fluid.layers.elementwise_min(fluid.layers.elementwise_max(negpos_ratio * num_pos, fluid.layers.zeros([1], dtype='int32')), fluid.layers.shape(pos)[1]-1)
        # fluid.layers.py_func(func=debug_sum, x=num_neg ,out=None)
        neg = idx_rank < fluid.layers.expand_as(num_neg, idx_rank)
        
        neg = fluid.layers.cast(neg, 'int32')
        neg *= -(pos - 1) 
        neg *= -(fluid.layers.cast(conf_t < 0, 'int32') - 1)        
        
        
        posneg = (pos + neg) > 0
        
        # posneg = pos > 0
        
        out = fluid.layers.where(posneg)
        out.stop_gradient = True
        conf_data_t = fluid.layers.gather_nd(conf_data, out)
        conf_tt = fluid.layers.gather_nd(conf_t, out)
        conf_tt = fluid.layers.reshape(conf_tt, shape=(-1,1))
        
        
        # conf_data_t.stop_gradient=True
        conf_tt.stop_gradient=True
        # fluid.layers.py_func(func=debug_gt, x=conf_data_t ,out=None)  
        # fluid.layers.py_func(func=debug_max, x=conf_tt ,out=None) 
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=fluid.layers.cast(conf_data_t,'float32'), 
            label=fluid.layers.cast(conf_tt,'int64'))

        loss = fluid.layers.reduce_sum(loss)
        # fluid.layers.py_func(func=debug_gt, x=loss ,out=None)  
        # fluid.layers.py_func(func=debug_gt, x=t ,out=None) 
        return loss
        
    def lincomb_mask_loss(self, pos, idx_t, mask_data, proto_data, masks, gt_box_t, labels, gt_num, batch_size, num_priors, interpolation_mode='bilinear'):
        mask_h = 138
        mask_w = 138
        
        flag = create_tmp_var('flag','bool',[1])
        # fluid.layers.py_func(func=debug_gt, x=flag ,out=None)
        maskiou_t_list = []
        maskiou_net_input_list = []
        label_t_list = []
        
        for idx in range(batch_size):
            
            downsampled_masks = fluid.layers.squeeze(
                                    fluid.layers.resize_bilinear(
                                        fluid.layers.unsqueeze(input=masks[idx], axes=[0]), 
                                        out_shape=[mask_h, mask_w], align_corners=False),
                                    axes=[0])
            # downsampled_masks = fluid.layers.transpose(downsampled_masks, perm=[1, 2, 0])
            downsampled_masks = fluid.layers.cast(downsampled_masks > 0.5,'float32')
            
            
            cur_pos = fluid.layers.cast(pos,'int32')[idx]
            out = fluid.layers.where(fluid.layers.cast(cur_pos,'bool'))
            out.stop_gradient = True
            # idx_t.stop_gradient=True
            # gt_box_t.stop_gradient=True
            pos_idx_t = fluid.layers.gather_nd(idx_t[idx], out)
            pos_gt_box_t = fluid.layers.gather_nd(gt_box_t[idx], out)
            # fluid.layers.py_func(func=debug_gt, x=out ,out=None)
            
            # fluid.layers.py_func(func=iszero, x=pos_idx_t ,out=flag)
            
            # if flag:
            #     continue
            
            proto_masks = proto_data[idx]
            proto_coef = fluid.layers.gather_nd(mask_data[idx], out)
            
            old_num_pos = fluid.layers.shape(proto_coef)[0]
            select = create_tmp_var('select','int32',[-1,1])
            fluid.layers.py_func(func=get_mast, x=proto_coef ,out=select)
            masks_to_train = fluid.layers.fill_constant(shape=[1], dtype='int32', value=100)
            out = fluid.layers.cond(old_num_pos > masks_to_train, lambda: masks_to_train, lambda: old_num_pos)
            select = select[:out,]
            
            select.stop_gradient=True
            
            
            proto_coef = fluid.layers.gather(proto_coef, select, overwrite=True)
            pos_idx_t = fluid.layers.gather(pos_idx_t, select, overwrite=True)
            pos_gt_box_t = fluid.layers.gather(pos_gt_box_t, select, overwrite=True)
            # fluid.layers.py_func(func=debug_max, x=pos_idx_t ,out=None)
            
            pos_idx_t.stop_gradient=True
            pos_gt_box_t.stop_gradient=True            
            downsampled_masks.stop_gradient=True
            downsampled_masks = fluid.layers.gather(downsampled_masks, pos_idx_t, overwrite=True)
            label_t = fluid.layers.gather(labels[idx], pos_idx_t, overwrite=True)
            # fluid.layers.py_func(func=debug_max, x=label_t ,out=None) 
            mask_t = fluid.layers.transpose(downsampled_masks, perm=[1, 2, 0])
            # fluid.layers.py_func(func=debug_gt, x=mask_t ,out=None) 
            
            # proto_coef.stop_gradient=True
            pred_masks = fluid.layers.matmul(proto_masks, proto_coef, transpose_x=False, transpose_y=True)
            pred_masks = fluid.layers.sigmoid(pred_masks)
            
            pred_masks = crop_tensor(pred_masks, pos_gt_box_t)
            
            mask_t.stop_gradient=True
            pred_masks = fluid.layers.clip(pred_masks, 0, 1)
            pre_loss = - (fluid.layers.log(pred_masks+1e-10) * mask_t + (1 - mask_t) * fluid.layers.log(1 - pred_masks+1e-10))
            pre_loss = crop_tensor(pre_loss, pos_gt_box_t)
            pred_masks = crop_tensor(pred_masks, pos_gt_box_t)

            weight = mask_h * mask_w
            
            gt_box_width, gt_box_height = get_box_height_width_tensor(pos_gt_box_t, mask_w, mask_h)
            gt_box_width.stop_gradient=True
            gt_box_height.stop_gradient=True
            
            pre_loss = fluid.layers.reduce_sum(pre_loss, dim=[0, 1]) / (gt_box_width) / (gt_box_height) * weight
            
            num_pos = fluid.layers.shape(proto_coef)[0]
            rate = old_num_pos / num_pos
            rate.stop_gradient = True
            
            if idx == 0:
                loss_m = fluid.layers.reduce_sum(pre_loss) * fluid.layers.cast(rate, 'float32')
            else:
                loss_m += fluid.layers.reduce_sum(pre_loss) * fluid.layers.cast(rate, 'float32')
            
            gt_mask_area = fluid.layers.reduce_sum(mask_t, dim=[0, 1])
            discard_mask_area = 25
            new_select = gt_mask_area > discard_mask_area
            out = fluid.layers.where(new_select)
            one = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
            pred = fluid.layers.less_than(fluid.layers.shape(out)[0], one) 
            out1 = fluid.layers.reshape(fluid.layers.range(0, fluid.layers.shape(pos_gt_box_t)[0], 1, 'int32'), (-1,1))
            # fluid.layers.py_func(func=debug_shape, x=out1 ,out=None) 
            #静态图判断不太会，只能这么代替了
            out_res = fluid.layers.cond(pred, lambda: out1, lambda: out)
            # fluid.layers.py_func(func=debug_shape, x=out_res ,out=None) 
            
            out_res.stop_gradient = True
            pos_gt_box_t = fluid.layers.gather(pos_gt_box_t, out_res)
            pred_masks = fluid.layers.transpose(pred_masks, perm=[2, 0, 1])
            pred_masks = fluid.layers.gather(pred_masks, out_res)
            mask_t = fluid.layers.transpose(mask_t, perm=[2, 0, 1])
            mask_t = fluid.layers.gather(mask_t, out_res)
            mask_t = fluid.layers.transpose(mask_t, perm=[1, 2, 0])
            label_t = fluid.layers.gather(label_t, out_res)

            maskiou_net_input = fluid.layers.unsqueeze(input=pred_masks, axes=[1])
            pred_masks = fluid.layers.transpose(pred_masks, perm=[1, 2, 0])
            pred_masks = fluid.layers.cast(pred_masks > 0.5,'float32')
            maskiou_t = self._mask_iou(pred_masks, mask_t)
            
            maskiou_net_input_list.append(maskiou_net_input)
            maskiou_t_list.append(maskiou_t)
            label_t_list.append(label_t)
            
        mask_alpha = 6.125
        # loss_m.stop_gradient=True
        losses = loss_m * mask_alpha / mask_h / mask_w
        
        # fluid.layers.py_func(func=debug_gt, x=loss_m ,out=None) 
        # # if len(maskiou_t_list) == 0:
        # #     return losses, None      
        
        maskiou_t = fluid.layers.concat(maskiou_t_list, axis=0)
        label_t = fluid.layers.concat(label_t_list, axis=0)
        maskiou_net_input = fluid.layers.concat(maskiou_net_input_list, axis=0) 
        # fluid.layers.py_func(func=debug_max, x=label_t ,out=None) 
        return losses, [maskiou_net_input, maskiou_t, label_t]
        
    def _mask_iou(self, mask1, mask2):
        intersection = fluid.layers.reduce_sum(mask1*mask2, dim=[0, 1])
        area1 = fluid.layers.reduce_sum(mask1, dim=[0, 1])
        area2 = fluid.layers.reduce_sum(mask2, dim=[0, 1])
        union = (area1 + area2) - intersection
        ret = intersection / (union + 1e-10)
        return ret        
        
    def judge_discard_mask(self, new_select, pred_masks, pos_gt_box_t, mask_t, label_t):
        out = fluid.layers.where(new_select)
        one = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        pred = fluid.layers.less_than(fluid.layers.shape(out)[0], one)      
        
        def noop():
            pred_masks_output = fluid.layers.transpose(pred_masks, perm=[2, 0, 1])
            fluid.layers.assign(input=pred_masks_output, output=pred_masks)
                
        def select_right():
            pos_gt_box_t_output = fluid.layers.gather(pos_gt_box_t, out)
            pred_masks_output = fluid.layers.transpose(pred_masks, perm=[2, 0, 1])
            pred_masks_output = fluid.layers.gather(pred_masks_output, out)
            mask_t_output = fluid.layers.transpose(mask_t, perm=[2, 0, 1])
            mask_t_output = fluid.layers.gather(mask_t_output, out)
            mask_t_output = fluid.layers.transpose(mask_t_output, perm=[1, 2, 0])
            label_t_output = fluid.layers.gather(label_t, out)
            fluid.layers.assign(input=pred_masks_output, output=pred_masks)
            fluid.layers.assign(input=pos_gt_box_t_output, output=pos_gt_box_t)
            fluid.layers.assign(input=mask_t_output, output=mask_t)
            fluid.layers.assign(input=label_t_output, output=label_t)
            
        fluid.layers.cond(pred, noop, select_right)
        
        return pred_masks, pos_gt_box_t, mask_t, label_t
    
def debug_sum(x):
    x = np.array(x)
    print(np.shape(x),np.sum(x))

def debug_min(x):
    x = np.array(x)
    print('min',np.min(x))

def debug_max(x):
    x = np.array(x)
    print('max',np.max(x))

def transform_index(idx):
    idx = np.array(idx)
    res = []
    for i in range(len(idx)):
        res.append(np.array([i,idx[i]]))
    return np.array(res)

def debug_shape(x):
    x = np.array(x)
    print('debug_shape:',np.shape(x))

def get_select(gt_mask_area):
    gt_mask_area = np.array(gt_mask_area)
    #########
    discard_mask_area = 10
    select = gt_mask_area > discard_mask_area
    # print('select',select)
    return select

def get_box_height_width_tensor(pos_gt_box_t, mask_w, mask_h):
    pos_gt_csize = center_size_tensor(pos_gt_box_t)
    gt_box_width  = pos_gt_csize[:, 2] * mask_w
    gt_box_height = pos_gt_csize[:, 3] * mask_h
    return fluid.layers.cast(gt_box_width, 'float32'), fluid.layers.cast(gt_box_height, 'float32')
    
def center_size_tensor(boxes):
    return fluid.layers.concat(input=[ (boxes[:, 2:] + boxes[:, :2])/2,
                            boxes[:, 2:] - boxes[:, :2]  ], axis=1)


def get_box_height_width(pos_gt_box_t):
    mask_w = 144
    mask_h = 144
    pos_gt_box_t = np.array(pos_gt_box_t)
    pos_gt_csize = center_size(pos_gt_box_t)
    gt_box_width  = pos_gt_csize[:, 2] * mask_w
    gt_box_height = pos_gt_csize[:, 3] * mask_h
    # print('gt_box_width',np.shape(gt_box_width))
    return gt_box_width.astype('float32'), gt_box_height.astype('float32')

def get_mast_to_train(x):
    x = np.array(x)
    masks_to_train = 100
    if np.shape(x)[0] > masks_to_train:
       perm = np.random.permutation(np.shape(x)[0]) 
       select = perm[:masks_to_train]
       return select
    return np.random.permutation(np.shape(x)[0])
    
def get_mast(x):
    x = np.array(x)
    
    perm = np.random.permutation(np.shape(x)[0])
    return perm
    # return np.reshape(perm[:masks_to_train], (-1,1))
    
def iszero(x):
    x = np.array(x)
    # print('xlen:',np.shape(x)[0])
    return True if np.shape(x)[0] == 0 else False

def get_cur_pos(cur_pos):
    cur_pos = np.array(cur_pos)
    return np.reshape(cur_pos, (-1,1))

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return np.concatenate(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h    

def assign_segment(segment_t, cur_class_t, cur_num, downsampled_masks):
    segment_t = np.array(segment_t)
    cur_class_t = np.array(cur_class_t)
    cur_num = np.array(cur_num)[0]
    downsampled_masks = np.array(downsampled_masks)
    
    for obj_idx in range(cur_num):
        segment_t[cur_class_t[obj_idx]-1] = np.maximum(segment_t[cur_class_t[obj_idx]-1], downsampled_masks[obj_idx])
        
    return segment_t      

def get_segment_t_tensor(mask_t, cur_class_t, cur_num, mask_w, mask_h):
    downsampled_masks = fluid.layers.squeeze(
                            fluid.layers.resize_bilinear(
                                fluid.layers.unsqueeze(input=mask_t, axes=[0]), 
                                out_shape=[mask_w, mask_h],
                                align_corners=False),
                            axes=[0]) 
    downsampled_masks = fluid.layers.cast(downsampled_masks > 0.5, 'float32')
    
    segment_t = fluid.layers.zeros(shape=[80, mask_w, mask_h], dtype='float32')
    
    fluid.layers.py_func(func=assign_segment, x=[segment_t, cur_class_t, cur_num, downsampled_masks] ,out=segment_t)
    
    return segment_t
    
    
            
def get_segment_t(mask_t, cur_class_t, cur_num):
    
    mask_t = np.array(mask_t)
    cur_class_t = np.array(cur_class_t)
    cur_num = np.array(cur_num)[0]
    
    downsampled_masks = np.zeros((cur_num, 72, 72))
    for idx in range(cur_num):
        temp = mask_t[idx]
        downsampled_masks[idx] = cv2.resize(temp, (72, 72))
        
    downsampled_masks = (downsampled_masks > 0.5).astype('float32')
    
    segment_t = np.zeros((80, 72, 72),dtype='float32')
    
    for obj_idx in range(cur_num):
        segment_t[cur_class_t[obj_idx]-1] = numpyminmax(segment_t[cur_class_t[obj_idx]-1], downsampled_masks[obj_idx])
        
    return segment_t    

def get_posneg(conf_data, conf_t, pos):
    conf_data = np.array(conf_data)
    batch_size = np.shape(conf_data)[0]
    conf_t = np.array(conf_t)
    pos = np.array(pos)
    
    # print('pos start',np.sum(pos))
    
    # Compute max conf across batch for hard negative mining
    batch_conf = np.reshape(conf_data, (-1, 81))
    loss_c = log_sum_exp(batch_conf) - np.reshape(batch_conf[:, 0],(-1,1))
    # print('loss_c:',loss_c)
    
    # Hard Negative Mining
    loss_c = np.reshape(loss_c, (batch_size, -1))
    loss_c[pos] = 0
    loss_c[conf_t < 0] = 0
    
    loss_idx = np.argsort(-loss_c, 1)
    idx_rank = np.argsort(loss_idx, 1)
    num_pos = np.sum(pos, 1, keepdims=True)
    negpos_ratio = 3
    num_neg = np.clip(negpos_ratio * num_pos, a_min=0, a_max=np.shape(pos)[1]-1)
    neg = idx_rank < np.broadcast_to(num_neg,np.shape(idx_rank))
    
    neg[pos] = 0
    neg[conf_t < 0] = 0
    
    pos_idx = np.broadcast_to(np.expand_dims(pos, 2),np.shape(conf_data))
    neg_idx = np.broadcast_to(np.expand_dims(neg, 2),np.shape(conf_data))
    
    posneg_idx = np.add(pos_idx, neg_idx)
    posneg = np.add(pos, neg)
    # print('posneg',np.sum(posneg))
    # print('posneg shape:',np.shape(posneg))
    # print('posneg:',posneg)
    posneg_idx = np.reshape(posneg_idx,(-1,1))
    # posneg = np.reshape(posneg,(-1,1))
    
    # print('get_posneg------------------------------------------------------------------------------------------------------------------------------------------------')
    
    return posneg, np.reshape(posneg,(-1,1)), posneg_idx
    
def log_sum_exp(x):
    x_max = x.max()
    return np.log(np.exp(x-x_max).sum(1,keepdims=True)) + x_max

start_t = 0

def start_time():
    global start_t 
    start_t = time.clock()

def end_time():
    global start_t
    print('time2',time.clock() - start_t)

def crop_tensor(masks, boxes):
    # fluid.layers.py_func(func=start_time, x=fluid.layers.shape(masks) ,out=None)
    padding = 1
    s = fluid.layers.shape(masks)
    h = fluid.layers.cast(s[0], 'float32')
    w = fluid.layers.cast(s[1], 'float32')
    n = fluid.layers.cast(s[2], 'float32')
    x1, x2 = sanitize_coordinates_tensor(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates_tensor(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = fluid.layers.expand_as(fluid.layers.reshape(fluid.layers.range(0, w, 1, 'float32'), shape=(1, -1, 1)), target_tensor=masks)
    cols = fluid.layers.expand_as(fluid.layers.reshape(fluid.layers.range(0, h, 1, 'float32'), shape=(-1, 1, 1)), target_tensor=masks)
    # fluid.layers.py_func(func=debug_sum, x=rows ,out=None)
    
    masks_left  = rows >= fluid.layers.reshape(x1, shape=(1,1,-1))
    masks_right = rows <  fluid.layers.reshape(x2, shape=(1,1,-1))
    masks_up    = cols >= fluid.layers.reshape(y1, shape=(1,1,-1))
    masks_down  = cols <  fluid.layers.reshape(y2, shape=(1,1,-1))
    
    masks_left  = fluid.layers.cast(masks_left, 'float32')
    masks_right = fluid.layers.cast(masks_right, 'float32')
    masks_up    = fluid.layers.cast(masks_up, 'float32')
    masks_down  = fluid.layers.cast(masks_down, 'float32')
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    # fluid.layers.py_func(func=end_time, x=fluid.layers.shape(masks) ,out=None)
    crop_mask.stop_gradient = True
    # fluid.layers.py_func(func=debug_sum, x=crop_mask ,out=None) 
    return masks * crop_mask
    
def sanitize_coordinates_tensor(_x1, _x2, img_size, padding:int=0, cast:bool=True, is_mask=True):
    _x1 = fluid.layers.elementwise_mul(fluid.layers.cast(_x1, 'float32'), img_size)
    _x2 = fluid.layers.elementwise_mul(fluid.layers.cast(_x2, 'float32'), img_size)
    if cast:
        _x1 = fluid.layers.cast(_x1, 'int32')
        _x2 = fluid.layers.cast(_x2, 'int32')
    x1 = fluid.layers.elementwise_min(_x1, _x2)
    x2 = fluid.layers.elementwise_max(_x1, _x2)
    x1 = fluid.layers.clip(x=x1-padding, min=0, max=10000)
    if is_mask:
        x2 = fluid.layers.clip(x=x2+padding, min=-10000, max=138)
    else:
        x2 = fluid.layers.clip(x=x2+padding, min=-10000, max=550)
    
    return x1, x2

def crop(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    start_time = time.time()
    padding = 1
    masks = np.array(masks)
    boxes = np.array(boxes)
    h, w, n = np.shape(masks)
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)
    rows = np.broadcast_to(np.reshape(np.arange(w, dtype=x1.dtype),(1, -1, 1)),(h, w, n))
    cols = np.broadcast_to(np.reshape(np.arange(h, dtype=x1.dtype),(-1, 1, 1)),(h, w, n))
    # print('rows',np.sum(rows))
    
    masks_left  = rows >= np.reshape(x1, (1, 1, -1))
    masks_right = rows <  np.reshape(x2, (1, 1, -1))
    masks_up    = cols >= np.reshape(y1, (1, 1, -1))
    masks_down  = cols <  np.reshape(y2, (1, 1, -1))
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    
    # print('crop_mask:',np.sum(crop_mask))
    end_time = time.time()
    # print('time1',end_time - start_time)
    return crop_mask.astype('float32')

def sanitize_coordinates(_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.astype('int32')
        _x2 = _x2.astype('int32')
    x1 = numpyminmax(_x1, _x2, False)
    x2 = numpyminmax(_x1, _x2)
    x1 = np.clip(x1-padding, a_min=0, a_max=1000000)
    x2 = np.clip(x2+padding, a_min=-1000000, a_max=img_size)

    return x1, x2 
    
def numpyminmax(x,y,Max=True):
    if Max:
        return x*(x>y)+y*(x<y)
    else:
        return x*(y>x)+y*(y<x)

def get_target_tensor(gt_box, priors, gt_class, is_crowd, gt_num, loc_data, batch_size, num_priors):
    loc_t = []
    gt_box_t = []
    conf_t = []
    idx_t = []
    
    labels = []
        
    for idx in range(batch_size):
        num = gt_num[idx]
        truths = gt_box[idx, 0:num]
        labels.append(gt_class[idx, 0:num])
        
        crowd_boxes = None
        pos_threshold = 0.5
        neg_threshold = 0.4        
        loc, conf, best_truth_idx = match_tensor(pos_threshold, neg_threshold,
                  truths, priors, labels[idx], crowd_boxes,
                  loc_t, conf_t, idx_t, idx)  
        
        loc_t.append(loc)
        conf_t.append(conf)
        idx_t.append(best_truth_idx)
        gt_box_t.append(fluid.layers.gather(truths, idx_t[idx]))
    
    
    loc_t = fluid.layers.stack(loc_t, 0)
    gt_box_t = fluid.layers.stack(gt_box_t, 0)
    conf_t = fluid.layers.stack(conf_t, 0)
    idx_t = fluid.layers.stack(idx_t, 0)

    # loc_t = fluid.layers.concat(loc_t, 0)
    # loc_t = fluid.layers.reshape(loc_t, (batch_size, num_priors, 4))
    # gt_box_t = fluid.layers.concat(gt_box_t, 0)
    # gt_box_t = fluid.layers.reshape(gt_box_t, (batch_size, num_priors, 4))
    # conf_t = fluid.layers.concat(conf_t, 0)
    # conf_t = fluid.layers.reshape(conf_t, (batch_size, num_priors))
    # idx_t = fluid.layers.concat(idx_t, 0)
    # idx_t = fluid.layers.reshape(idx_t, (batch_size, num_priors))
    
    pos = conf_t > 0
    
    return loc_t, gt_box_t, conf_t, idx_t, pos
        
def transform_conf(conf, best_truth_overlap):
    conf = np.array(conf)
    best_truth_overlap = np.array(best_truth_overlap)
    pos_thresh = 0.5
    neg_thresh = 0.4
    conf[best_truth_overlap < pos_thresh] = -1 
    conf[best_truth_overlap < neg_thresh] =  0 
    return conf.astype('int32')

def assign_labels(overlaps, best_truth_overlap, best_truth_idx):
    overlaps = np.array(overlaps)
    best_truth_overlap = np.array(best_truth_overlap)
    best_truth_idx = np.array(best_truth_idx)
    for _ in range(np.shape(overlaps)[0]):
        best_prior_overlap = overlaps.max(1)
        best_prior_idx = overlaps.argmax(1)
        j = best_prior_overlap.argmax(0)
        i = best_prior_idx[j]
        overlaps[:, i] = -1
        overlaps[j, :] = -1
        best_truth_overlap[i] = 2
        best_truth_idx[i] = j   
    return best_truth_overlap, best_truth_idx
        
def match_tensor(pos_thresh, neg_thresh, truths, priors, labels, crowd_boxes, loc_t, conf_t, idx_t, idx):
    use_yolo_regressors = False
    decoded_priors = point_form_tensor(priors)
    
    #shape:[gt_num, num_priors]
    overlaps = jaccard_tensor(truths, decoded_priors)
    best_truth_overlap, best_truth_idx = fluid.layers.argsort(overlaps, 0, descending=True)
    best_truth_overlap = best_truth_overlap[0]
    # fluid.layers.py_func(func=debug_gt, x=best_truth_overlap ,out=None)
    best_truth_idx = best_truth_idx[0]
    # fluid.layers.py_func(func=debug_gt, x=best_truth_idx ,out=None)
    
    fluid.layers.py_func(func=assign_labels, x=[overlaps, best_truth_overlap, best_truth_idx] ,out=[best_truth_overlap, best_truth_idx])
    
        
    matches = fluid.layers.gather(truths, best_truth_idx)
    conf = fluid.layers.gather(labels, best_truth_idx)
    fluid.layers.py_func(func=transform_conf, x=[conf, best_truth_overlap] ,out=conf)
      
    # crowd_iou_threshold = 0.7
    # if crowd_boxes is not None:
    #     crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
    #     best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
    #     conf[(conf <= 0) & (best_crowd_overlap > crowd_iou_threshold)] = -1
        
    loc = encode_tensor(matches, priors, use_yolo_regressors)
    
    return loc, conf, best_truth_idx    
    
def decode(loc, priors):
    variances = [0.1, 0.2]
        
    boxes = np.concatenate([
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])], 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    
    return boxes  
    
def encode_tensor(matched, priors, use_yolo_regressors:bool=False):
    variances = [0.1, 0.2]

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2.0 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = fluid.layers.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    loc = fluid.layers.concat([g_cxcy, g_wh], 1)  # [num_priors,4]
    
    return loc

def jaccard_tensor_3D(box_a, box_b, iscrowd:bool=False):
    use_batch = True
    
    A = fluid.layers.shape(box_a)[1]
    B = fluid.layers.shape(box_b)[1]      
    
    inter = intersect_tensor(box_a, box_b)
    area_a = fluid.layers.expand(fluid.layers.unsqueeze(((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])),2), [1, 1, B])
    area_b = fluid.layers.expand(fluid.layers.unsqueeze(((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])),1), [1, A, 1])
    union = area_a + area_b - inter              

    out = inter / (area_a) if iscrowd else inter / (union)
    return out if use_batch else fluid.layers.squeeze(out, [0]) 

def jaccard_tensor(box_a, box_b, iscrowd:bool=False):
    use_batch = False
    box_a = fluid.layers.unsqueeze(box_a, [0])
    box_b = fluid.layers.unsqueeze(box_b, [0])
    
    A = fluid.layers.shape(box_a)[1]
    B = fluid.layers.shape(box_b)[1]      
    
    inter = intersect_tensor(box_a, box_b)
    area_a = fluid.layers.expand(fluid.layers.unsqueeze(((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])),2), [1, 1, B])
    area_b = fluid.layers.expand(fluid.layers.unsqueeze(((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])),1), [1, A, 1])
    union = area_a + area_b - inter              

    out = inter / (area_a) if iscrowd else inter / (union)
    return out if use_batch else fluid.layers.squeeze(out, [0])    

def intersect_tensor(box_a, box_b):
    
    n = fluid.layers.shape(box_a)[0]
    A = fluid.layers.shape(box_a)[1]
    B = fluid.layers.shape(box_b)[1]   
    
    # target_shape = fluid.layers.zeros(shape=[n, A, B, 2], dtype='int32')
    
    max_xy = fluid.layers.elementwise_min(fluid.layers.expand(fluid.layers.unsqueeze(box_a[:, :, 2:], 2), [1, 1, B, 1]),
                                    fluid.layers.expand(fluid.layers.unsqueeze(box_b[:, :, 2:], 1), [1, A, 1, 1]))
    min_xy = fluid.layers.elementwise_max(fluid.layers.expand(fluid.layers.unsqueeze(box_a[:, :, :2], 2), [1, 1, B, 1]),
                                    fluid.layers.expand(fluid.layers.unsqueeze(box_b[:, :, :2], 1), [1, A, 1, 1]))
    inter = fluid.layers.clip((max_xy - min_xy), min=0, max=1000)
    return inter[:, :, :, 0] * inter[:, :, :, 1]
    
    