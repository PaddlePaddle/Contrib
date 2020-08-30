#coding=utf-8
import getpass
import os
import socket
import numpy as np
from PIL import Image, ImageFilter
import argparse
import time
import sys
import pdb
import math

from utils import *
from dataset.dataset import *
from dataset.preprocess_data import *
from models.model import generate_model
from opts import parse_opts

import paddle
import paddle.fluid as fluid
    
if __name__=="__main__":
    opt = parse_opts()
    print(opt)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
        print("Preprocessing validation data ...")
        test_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 0, opt = opt)
        test_dataloader = paddle.batch(test_data, batch_size=opt.batch_size, drop_last=False)
        
        if opt.modality=='Flow': opt.input_channels = 2
        else: opt.input_channels = 3
    
        # Loading model and checkpoint
        model,_ = generate_model(opt)
        if opt.modality=='RGB' and opt.RGB_resume_path!='':
            para_dict, _ = fluid.dygraph.load_dygraph(opt.RGB_resume_path)
            model.set_dict(para_dict)
        if opt.modality=='Flow' and opt.Flow_resume_path!='':
            para_dict, _ = fluid.dygraph.load_dygraph(opt.Flow_resume_path)
            model.set_dict(para_dict)
        model.eval()
        accuracies = AverageMeter()
        clip_accuracies = AverageMeter()
        
        #Path to store results
        result_path = "{}/{}/".format(opt.result_path, opt.dataset)
        if not os.path.exists(result_path):
            os.makedirs(result_path)    

        for i, data in enumerate(test_dataloader()):
            #输入视频图像、光流
            # pdb.set_trace()
            clip = np.array([x[0] for x in data]).astype('float32')    
            # #输入视频图像、光流的标签       
            targets = np.array([x[1] for x in data]).astype('int')
            clip = np.squeeze(clip)
            if opt.modality == 'Flow':
                inputs = np.zeros((int(clip.shape[1]/opt.sample_duration), 2, opt.sample_duration, opt.sample_size, opt.sample_size),dtype=np.float32)
            else:
                inputs = np.zeros((int(clip.shape[1]/opt.sample_duration), 3, opt.sample_duration, opt.sample_size, opt.sample_size),dtype=np.float32)
            for k in range(inputs.shape[0]):
                inputs[k,:,:,:,:] = clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]  
            #将视频图像和光流分离开
            inputs = fluid.dygraph.base.to_variable(inputs)
            targets = fluid.dygraph.base.to_variable(targets)
            outputs= model(inputs)
            preds = fluid.layers.reduce_mean(outputs, dim=0, keep_dim=True)
            # pdb.set_trace()
            acc = calculate_accuracy(preds, targets) 
            accuracies.update(acc[0], targets.shape[0])            
        
        print("Video accuracy = ", accuracies.avg)