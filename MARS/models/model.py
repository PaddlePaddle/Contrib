#coding=utf-8
#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import division
import pdb

from models import resnext

import paddle
import paddle.fluid as fluid
from models.resnext import get_fine_tuning_parameters

def generate_model(opt):
    assert opt.model in ['resnext']
    assert opt.model_depth in [101]
    model = resnext.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            input_channels=opt.input_channels,
            output_layers=opt.output_layers)
    if opt.MARS_premodel_path != '' and opt.input_channels==3:
        print('loading pretrained model {}'.format(opt.MARS_premodel_path))
        para_dict, _ = fluid.dygraph.load_dygraph(opt.MARS_premodel_path)
        #设置网络模型参数为读取的模型参数
        model.set_dict(para_dict)
        model.fc = fluid.dygraph.Linear(model.lastfeature_size, opt.n_finetune_classes,
                                param_attr=fluid.ParamAttr(initializer=fluid.initializer.MSRAInitializer(uniform=True)), 
                                bias_attr=paddle.fluid.ParamAttr(initializer=None),
                                act="softmax")
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model,parameters
    elif  opt.Flow_premodel_path != '' and opt.input_channels==2:
        print('loading pretrained model {}'.format(opt.Flow_premodel_path))
        para_dict, _ = fluid.dygraph.load_dygraph(opt.Flow_premodel_path)
        #设置网络模型参数为读取的模型参数
        model.set_dict(para_dict)
        model.fc = fluid.dygraph.Linear(model.lastfeature_size, opt.n_finetune_classes,
                                param_attr=fluid.ParamAttr(initializer=fluid.initializer.MSRAInitializer(uniform=True)), 
                                bias_attr=paddle.fluid.ParamAttr(initializer=None),
                                act="softmax")
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model,parameters
    elif opt.RGB_premodel_path != '' and opt.input_channels==3:
        print('loading pretrained model {}'.format(opt.RGB_premodel_path))
        para_dict, _ = fluid.dygraph.load_dygraph(opt.RGB_premodel_path)
        #设置网络模型参数为读取的模型参数
        model.set_dict(para_dict)
        model.fc = fluid.dygraph.Linear(model.lastfeature_size, opt.n_finetune_classes,
                                param_attr=fluid.ParamAttr(initializer=fluid.initializer.MSRAInitializer(uniform=True)), 
                                bias_attr=paddle.fluid.ParamAttr(initializer=None),
                                act="softmax")
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model,parameters
    return model,model.parameters()

