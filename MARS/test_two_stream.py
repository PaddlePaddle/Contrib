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

from utils import *
from dataset.dataset import *
from dataset.preprocess_data import *
from models.model import generate_model
from opts import parse_opts

import paddle
import paddle.fluid as fluid
    
if __name__=="__main__":
    # print configuration options
    opt = parse_opts()
    print(opt)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
        print("Preprocessing validation data ...")
        val_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt)
        val_dataloader = paddle.batch(val_data, batch_size=opt.batch_size, drop_last=True)
        
        opt.input_channels = 3
        model1 = generate_model(opt)
        if opt.MARS_resume_path:
            para_dict, _ = fluid.dygraph.load_dygraph(opt.MARS_resume_path)
            model1.set_dict(para_dict)
        model1.eval()
        
        opt.input_channels = 2
        model2 = generate_model(opt)
        if opt.Flow_resume_path:
            para_dict, _ = fluid.dygraph.load_dygraph(opt.Flow_resume_path)
            model2.set_dict(para_dict)
        model2.eval()
        
        opt.input_channels = 3
        model3 = generate_model(opt)
        if opt.RGB_resume_path:
            para_dict, _ = fluid.dygraph.load_dygraph(opt.RGB_resume_path)
            model3.set_dict(para_dict)
        model3.eval()
        
        accuracies = AverageMeter()
        result_path = "{}/{}/".format(opt.result_path, opt.dataset)
        if not os.path.exists(result_path):
            os.makedirs(result_path)    

        if opt.log:
            f = open(os.path.join(result_path, "test_{}{}_{}_{}_{}_{}.txt".format( opt.model, opt.model_depth, opt.dataset, opt.split, opt.modality, opt.sample_duration)), 'w+')
            f.write(str(opt))
            f.write('\n')
            f.flush()
        for i, data in enumerate(val_dataloader()):
            #输入视频图像、光流
            inputs = np.array([x[0] for x in data]).astype('float32')    
            #输入视频图像、光流的标签       
            targets = np.array([x[1] for x in data]).astype('int')
            #将视频图像和光流分离开
            inputs1 = inputs[:,0:3,:,:,:]
            inputs2 = inputs[:,3:,:,:,:]
            inputs1 = fluid.dygraph.base.to_variable(inputs1)
            inputs2 = fluid.dygraph.base.to_variable(inputs2)
            targets = fluid.dygraph.base.to_variable(targets)
            outputs1= model1(inputs1)
            outputs2= model2(inputs2)
            outputs3= model3(inputs1)
            outputs=fluid.layers.concat([outputs1,outputs2,outputs3],axis=0)
            outputs=fluid.layers.reduce_mean(outputs,dim=0)
            acc = calculate_accuracy(outputs, targets)   
            accuracies.update(acc[0], targets.shape[0])            
            line = "Video[" + str(i) + "] : top1 = " + str(acc[0]) +  "\t  acc = " + str(accuracies.avg)
            print(line)
            if opt.log:
                f.write(line + '\n')
                f.flush()
        
        print("Video accuracy = ", accuracies.avg)
        line = "Video accuracy = " + str(accuracies.avg) + '\n'
        if opt.log:
            f.write(line)