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

import getpass
import os
import socket
import numpy as np
from PIL import Image, ImageFilter
import argparse

import time
import sys
#from utils import AverageMeter, calculate_accuracy
import pdb
import math

from dataset.dataset import *
from dataset.preprocess_data import *
from models.model import generate_model
from opts import parse_opts
from utils import *
import pdb
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.learning_rate_scheduler import ReduceLROnPlateau

def resume_params(model, optimizer, opt):
    """
    加载模型参数
    参数：
    model，定义的网络模型
    optimizer，网络优化器
    opt，配置参数
    :return:
    如果有之前保存的checkpoint，从之前的checkpoint恢复模型参数
    """
    if opt.continue_train and os.path.exists(opt.Flow_resume_path):
        print("you now read checkpoint!!!")
        checkpoint_list=os.listdir(opt.Flow_resume_path)
        max_epoch=0
        for checkpoint in checkpoint_list:
            if 'model_Flow_' in checkpoint:
                max_epoch=max(int(checkpoint.split('_')[2]),max_epoch)
        if max_epoch>0:
            #从checkpoint读取模型参数和优化器参数
            para_dict, opti_dict = fluid.dygraph.load_dygraph(os.path.join(opt.Flow_resume_path,'model_Flow_'+str(max_epoch)+'_saved'))
            #设置网络模型参数为读取的模型参数
            model.set_dict(para_dict)
            #设置优化器参数为读取的优化器参数
            optimizer.set_dict(opti_dict)
            #更新当前网络的开始迭代次数
            opt.begin_epoch=max_epoch+1

def train():
    #读取配置文件
    opt = parse_opts()
    print(opt)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    #
    with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
        #训练数据加载器
        print("Preprocessing train data ...")
        train_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 1, opt = opt)
        train_dataloader = paddle.batch(train_data, batch_size=opt.batch_size, drop_last=True)
        #训练数据加载器
        print("Preprocessing validation data ...")
        val_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt)
        val_dataloader = paddle.batch(val_data, batch_size=opt.batch_size, drop_last=True)
        
        #如果使用光流图像进行训练，输入通道数为2
        opt.input_channels = 2
         
        #构建网络模型结构
        print("Loading Flow model... ", opt.model, opt.model_depth)
        model,parameters = generate_model(opt)

        print("Initializing the optimizer ...")
        if opt.Flow_premodel_path: 
            opt.weight_decay = 1e-5
            opt.learning_rate = 0.001
            
        print("lr = {} \t momentum = {}, \t nesterov = {} \t LR patience = {} "
                    .format(opt.learning_rate, opt.momentum, opt.nesterov, opt.lr_patience))
        #构建优化器
        optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=opt.learning_rate, 
                    momentum=opt.momentum, parameter_list=parameters, 
                    use_nesterov=opt.nesterov)
        scheduler = ReduceLROnPlateau(opt.learning_rate, mode='min',  patience=opt.lr_patience)
        if opt.continue_train and opt.Flow_resume_path != '':
            resume_params(model, optimizer, opt)
        print('run')
        losses_avg=np.zeros((1,),dtype=np.float)
        for epoch in range(opt.begin_epoch, opt.n_epochs+1):
            #设置模型为训练模式，模型中的参数可以被训练优化
            model.train()
            batch_time = AverageMeter()
            data_time  = AverageMeter()
            losses     = AverageMeter()
            accuracies   = AverageMeter()
            end_time   = time.time()
            for i, data in enumerate(train_dataloader()):
                #输入视频图像或者光流
                inputs = np.array([x[0] for x in data]).astype('float32')
                # 输入视频图像或者光流的标签   
                targets = np.array([x[1] for x in data]).astype('int')
                inputs = fluid.dygraph.base.to_variable(inputs)
                targets = fluid.dygraph.base.to_variable(targets)
                targets.stop_gradient = True
                data_time.update(time.time() - end_time)
                #计算网络输出结果
                outputs = model(inputs)
                #计算网络输出和标签的交叉熵损失
                loss = fluid.layers.cross_entropy(outputs, targets)
                avg_loss = fluid.layers.mean(loss)
                #计算网络预测精度
                acc = calculate_accuracy(outputs, targets)
                losses.update(avg_loss.numpy()[0], inputs.shape[0])
                accuracies.update(acc[0], inputs.shape[0])
                #反向传播梯度
                optimizer.clear_gradients()
                avg_loss.backward()
                #最小化损失来优化网络中的权重
                #print(avg_loss)
                #pdb.set_trace()
                optimizer.minimize(avg_loss)
                batch_time.update(time.time() - end_time)
                end_time = time.time()
                
                print('Epoch: [{0}][{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss val:{loss.val:.4f} (avg:{loss.avg:.4f})\t'
                  'Acc val:{acc.val:.3f} (avg:{acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))
            losses_avg[0]=losses.avg
            scheduler.step(losses_avg)
            if epoch % opt.checkpoint == 0 and epoch != 0:
                fluid.dygraph.save_dygraph(model.state_dict(),os.path.join(opt.Flow_resume_path,'model_Flow_'+str(epoch)+'_saved'))
                fluid.dygraph.save_dygraph(optimizer.state_dict(), os.path.join(opt.Flow_resume_path,'model_Flow_'+str(epoch)+'_saved'))
            #设置模型为验证模式，对验证数据集进行验证
            model.eval()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            end_time = time.time()
            for i, data in enumerate(val_dataloader()):
                data_time.update(time.time() - end_time)
                inputs = np.array([x[0] for x in data]).astype('float32')           
                targets = np.array([x[1] for x in data]).astype('int')
                inputs = fluid.dygraph.base.to_variable(inputs)
                targets = fluid.dygraph.base.to_variable(targets)
                targets.stop_gradient = True
                outputs  = model(inputs)
                    
                loss = fluid.layers.cross_entropy(outputs, targets)
                avg_loss = fluid.layers.mean(loss)
                acc  = calculate_accuracy(outputs, targets)
    
                losses.update(avg_loss.numpy()[0], inputs.shape[0])
                accuracies.update(acc[0], inputs.shape[0])
                batch_time.update(time.time() - end_time)
                end_time = time.time()
                print('Val_Epoch: [{0}][{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                            epoch,
                            i + 1,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            acc=accuracies))

if __name__=="__main__":
    train()
