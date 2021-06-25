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
import paddle.nn as nn
import math
import paddle.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU


class PSENet(nn.Layer):
	def __init__(self,
	             backbone,
	             neck,
	             detection_head):
		super(PSENet, self).__init__()
		self.backbone = build_backbone(backbone)
		self.fpn = build_neck(neck)

		self.det_head = build_head(detection_head)

	def _upsample(self, x, size, scale=1):
		_, _, H, W = size
		return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

	def forward(self,
	            imgs,
	            gt_texts=None,
	            gt_kernels=None,
	            training_masks=None,
	            img_metas=None,
	            cfg=None):
		outputs = dict()

		if not self.training and cfg.report_speed:
			start = time.time()

		# backbone
		f = self.backbone(imgs)
		if not self.training and cfg.report_speed:
			outputs.update(dict(
				backbone_time=time.time() - start
			))
			start = time.time()

		# FPN
		f1, f2, f3, f4, = self.fpn(f[0], f[1], f[2], f[3])

		f = paddle.concat((f1, f2, f3, f4), 1)
		if not self.training and cfg.report_speed:
			outputs.update(dict(
				neck_time=time.time() - start
			))
			start = time.time()
		# detection

		det_out = self.det_head(f)
		if not self.training and cfg.report_speed:
			outputs.update(dict(
				det_head_time=time.time() - start
			))

		if self.training:
			det_out = self._upsample(det_out, imgs.shape)
			det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks)
			outputs.update(det_loss)
		else:
			det_out = self._upsample(det_out, imgs.shape, 1)
			det_res = self.det_head.get_results(det_out, img_metas, cfg)
			outputs.update(det_res)

		return outputs
