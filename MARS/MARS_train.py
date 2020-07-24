#coding=utf-8
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
    if opt.continue_train and os.path.exists(opt.MARS_resume_path):
        print("you now read checkpoint!!!")
        checkpoint_list=os.listdir(opt.MARS_resume_path)
        max_epoch=0
        for checkpoint in checkpoint_list:
            if 'model_MARS_' in checkpoint:
                max_epoch=max(int(checkpoint.split('_')[2]),max_epoch)
        if max_epoch>0:
            #从checkpoint读取模型参数和优化器参数
            para_dict, opti_dict = fluid.dygraph.load_dygraph(os.path.join(opt.MARS_resume_path,'model_MARS_'+str(max_epoch)+'_saved'))
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
    with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
        #训练数据加载函数
        print("Preprocessing train data ...")
        train_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 1, opt = opt)
        train_dataloader = paddle.batch(train_data, batch_size=opt.batch_size, drop_last=True)
        #验证数据加载函数
        print("Preprocessing validation data ...")
        val_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt)
        val_dataloader = paddle.batch(val_data, batch_size=opt.batch_size, drop_last=True)
        
        #日志文件设置
        log_path = os.path.join(opt.result_path, opt.dataset)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if opt.log == 1:
            if opt.MARS_premodel_path != '':
                epoch_logger = Logger_MARS(os.path.join(log_path, 'PreModel_MARS_{}_{}_train_batch{}_sample{}_model{}{}_ftbeginidx{}_layer{}.log'
                    .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.model, opt.model_depth, opt.ft_begin_index, opt.output_layers[0]))
                    ,['epoch', 'loss', 'loss_MSE', 'loss_MARS', 'acc'], opt.MARS_premodel_path, opt.begin_epoch)
                val_logger   = Logger_MARS(os.path.join(log_path, 'PreModel_MARS_{}_{}_val_batch{}_sample{}_model{}{}_ftbeginidx{}_layer{}.log'
                    .format(opt.dataset,opt.split,  opt.batch_size, opt.sample_size, opt.model, opt.model_depth, opt.ft_begin_index, opt.output_layers[0]))
                    ,['epoch', 'loss', 'acc'], opt.MARS_premodel_path, opt.begin_epoch)
            else:
                epoch_logger = Logger_MARS(os.path.join(log_path, 'MARS_{}_{}_train_batch{}_sample{}__model{}{}_ftbeginidx{}_layer{}.log'
                    .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.model, opt.model_depth, opt.ft_begin_index, opt.output_layers[0]))
                    ,['epoch', 'loss', 'loss_MSE', 'loss_MARS', 'acc'], opt.MARS_resume_path, opt.begin_epoch)
                val_logger   = Logger_MARS(os.path.join(log_path, 'MARS_{}_{}_val_batch{}_sample{}_model{}{}_ftbeginidx{}_layer{}.log'
                    .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.model, opt.model_depth, opt.ft_begin_index, opt.output_layers[0]))
                    ,['epoch', 'loss', 'acc'], opt.MARS_resume_path, opt.begin_epoch)
    
        
        #构建光流网络模型
        print("Loading Flow model... ", opt.model, opt.model_depth) 
        opt.input_channels =2 
        if opt.dataset == 'HMDB51':
            opt.n_classes = 51
        elif opt.dataset == 'Kinetics':
            opt.n_classes = 400 
        elif opt.dataset == 'UCF101':
            opt.n_classes = 101 
        model_Flow = generate_model(opt,'Flow')
        #如果光流部分网络有预定义模型，则加载预定义模型
        if opt.Flow_resume_path:
            print('loading Flow checkpoint {}'.format(opt.Flow_resume_path))
            para_dict, _ = fluid.dygraph.load_dygraph(opt.Flow_resume_path)
            model_Flow.set_dict(para_dict)
        
        #构建图像网络模型
        print("Loading MARS model... ", opt.model, opt.model_depth)
        opt.input_channels =3
        if opt.MARS_premodel_path != '':
            opt.n_classes = 400 
        model_MARS = generate_model(opt,'MARS')
        print("Initializing the optimizer ...")
        print("lr = {} \t momentum = {} \t weight_decay = {}, \t nesterov = {}"
            .format(opt.learning_rate, opt.momentum, opt. weight_decay, opt.nesterov))
        print("LR patience = ", opt.lr_patience) 
        #定义优化器
        optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=opt.learning_rate, 
                    momentum=opt.momentum, parameter_list=model_MARS.parameters(), 
                    use_nesterov=opt.nesterov)
        #pdb.set_trace()
        scheduler = ReduceLROnPlateau(opt.learning_rate, mode='min',  patience=opt.lr_patience)
        if opt.MARS_resume_path != '' and opt.continue_train:
            resume_params(model_MARS, optimizer, opt)
        print('run')
        #在网络训练过程中，光流网络参数保持固定
        model_Flow.train()
        losses_avg=np.zeros((1,),dtype=np.float)
        for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
            #设置模型为训练模式，模型中的参数可以被训练优化
            model_MARS.train()
            batch_time = AverageMeter()
            data_time  = AverageMeter()
            losses     = AverageMeter()
            losses_MARS = AverageMeter()
            losses_MSE = AverageMeter()
            accuracies   = AverageMeter()
            end_time   = time.time()
            
            for i, data in enumerate(train_dataloader()):
                #输入视频图像、光流
                inputs = np.array([x[0] for x in data]).astype('float32')    
                #输入视频图像、光流的标签       
                targets = np.array([x[1] for x in data]).astype('int')
                #将视频图像和光流分离开
                inputs_MARS = inputs[:,0:3,:,:,:]
                inputs_FLOW = inputs[:,3:,:,:,:]
                inputs_MARS = fluid.dygraph.base.to_variable(inputs_MARS)
                inputs_FLOW = fluid.dygraph.base.to_variable(inputs_FLOW)
                targets = fluid.dygraph.base.to_variable(targets)
                targets.stop_gradient = True
                data_time.update(time.time() - end_time)
                #计算图像的网络输出结果
                outputs_MARS  = model_MARS(inputs_MARS)
                #计算光流的网络输出结果
                outputs_Flow = model_Flow(inputs_FLOW)[1]
                #计算图像网络输出和标签的交叉熵损失
                loss_MARS = fluid.layers.cross_entropy(outputs_MARS[0], targets)
                loss_MARS = fluid.layers.mean(loss_MARS)
                #计算图像网络和光流网络提取特征的mse损失
                loss_MSE = opt.MARS_alpha*fluid.layers.mean(fluid.layers.mse_loss(outputs_MARS[1], outputs_Flow))
                
                #计算总的损失
                loss     = loss_MARS + loss_MSE
                optimizer.clear_gradients()
                #反向传播梯度
                loss.backward()
                #最小化损失来优化网络中的权重
                optimizer.minimize(loss)
                #计算网络预测精度
                acc = calculate_accuracy(outputs_MARS[0], targets)
                
                losses.update(loss.numpy()[0], inputs.shape[0])
                losses_MARS.update(loss_MARS.numpy()[0], inputs.shape[0])
                losses_MSE.update(loss_MSE.numpy()[0], inputs.shape[0])
                accuracies.update(acc[0], inputs.shape[0])
                batch_time.update(time.time() - end_time)
                end_time = time.time()
                print('Epoch: [{0}][{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lr {lr:.5f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_MARS {loss_MARS.val:.4f} ({loss_MARS.avg:.4f})\t'
                  'Loss_MSE {loss_MSE.val:.4f} ({loss_MSE.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      batch_time=batch_time,
                      data_time=data_time,
                      lr=optimizer.current_step_lr(),
                      loss=losses,
                      loss_MARS=losses_MARS,
                      loss_MSE=losses_MSE,
                      acc=accuracies))
            losses_avg[0]=losses.avg
            scheduler.step(losses_avg)
            if opt.log == 1:
                epoch_logger.log({
                    'epoch': epoch,
                    'loss': losses.avg,
                    'loss_MSE' : losses_MSE.avg,
                    'loss_MARS': losses_MARS.avg,
                    'acc': accuracies.avg
                })
            
            if epoch % opt.checkpoint == 0:
                fluid.dygraph.save_dygraph(model_MARS.state_dict(),os.path.join(opt.MARS_resume_path,'model_MARS_'+str(epoch)+'_saved'))
                fluid.dygraph.save_dygraph(optimizer.state_dict(), os.path.join(opt.MARS_resume_path,'model_MARS_'+str(epoch)+'_saved'))
            #设置图像网络模型为验证模式，对验证数据集进行验证
            model_MARS.eval()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            end_time = time.time()
            for i, data in enumerate(val_dataloader()):
                data_time.update(time.time() - end_time)
                inputs = np.array([x[0] for x in data]).astype('float32')           
                targets = np.array([x[1] for x in data]).astype('int')
                inputs_MARS = inputs[:,0:3,:,:,:]
                inputs_MARS = fluid.dygraph.base.to_variable(inputs_MARS)
                targets = fluid.dygraph.base.to_variable(targets)
                targets.stop_gradient = True
                
                outputs_MARS  = model_MARS(inputs_MARS)
                    
                loss = fluid.layers.cross_entropy(outputs_MARS[0], targets)
                acc  = calculate_accuracy(outputs_MARS[0], targets)
    
                losses.update(loss.numpy()[0], inputs.shape[0])
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
                              
            if opt.log == 1:
                val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
            
            




if __name__=="__main__":
    train()