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
import getpass
import os
import socket
import numpy as np
from PIL import Image, ImageFilter
import pickle
import glob
#import dircache
import pdb
from random import shuffle
from .preprocess_data import *

def get_test_video(opt, frame_path, Total_frames):
    """
    参数:
        opt         : 参数配置
        frame_path  : 视频帧的保存路径
        Total_frames: 总的视频帧数
    Returns:
        list(frames) : 所有视频帧的列表
    """

    clip = []
    i = 0
    loop = 0
    if Total_frames < opt.sample_duration:
        loop = 1
    #如果只使用RGB图像
    if opt.modality == 'RGB': 
        while len(clip) < max(opt.sample_duration, Total_frames):
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg'%(i+1)))
                clip.append(im.copy())
                im.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
    #如果只使用光流图像
    elif opt.modality == 'Flow':  
        while len(clip) < 2*max(opt.sample_duration, Total_frames):
            try:
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(i+1)))
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
    #如果使用RGB图像和光流图像 
    elif  opt.modality == 'RGB_Flow':
        while len(clip) < 3*max(opt.sample_duration, Total_frames):
            try:
                im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(i+1)))
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(i+1)))
                clip.append(im.copy())
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im.close()
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            if loop==1 and i == Total_frames:
                i = 0
    return clip

def get_train_video(opt, frame_path, Total_frames):
    """
    从训练或者验证视频中随机选择一些帧
    参数:
        opt         :参数配置
        frame_path  :视频帧的路径
        Total_frames:视频中的总帧数
    Returns:
        list(frames) : 从训练或者验证数据集的视频中随机选择的视频段帧
    """
    clip = []
    i = 0
    loop = 0
    #随机选择帧
    if Total_frames <= opt.sample_duration:#如果视频的总帧数小于要选择的帧数 
        loop = 1
        start_frame = np.random.randint(0, Total_frames)
    else:
        #随机从0到Total_frames - opt.sample_duration选择一个视频段的开始位置
        start_frame = np.random.randint(0, Total_frames - opt.sample_duration)
    #只使用RGB图像
    if opt.modality == 'RGB': 
        while len(clip) < opt.sample_duration:
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg'%(start_frame+i+1)))
                clip.append(im.copy())
                im.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
    #只使用光流图像
    elif opt.modality == 'Flow':  
        while len(clip) < 2*opt.sample_duration:
            try:
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(start_frame+i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(start_frame+i+1)))
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
    #使用RGB图像及光流图像       
    elif  opt.modality == 'RGB_Flow':
        while len(clip) < 3*opt.sample_duration:
            try:
                im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(start_frame+i+1)))
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(start_frame+i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(start_frame+i+1)))
                clip.append(im.copy())
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im.close()
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
    return clip



def HMDB51_test(train, opt, split=None):
    """
    参数:
        opt   : 参数配置
        train : 0表示测试, 1表示训练, 2表示验证 
        split : 1,2,3 
    Returns:
        (tensor(frames), class_id ):形状为C x T x H x W的张量及其标签
    """
    train_val_test = train
    lab_names = sorted(set(['_'.join(os.path.splitext(file)[0].split('_')[:-2]) for file in os.listdir(opt.annotation_path)]))
    #pdb.set_trace()
    #类别数目
    class_num = len(lab_names)
    print(class_num)
    assert class_num == 51
    lab_names = dict(zip(lab_names, range(class_num)))   # Each label is mappped to a number
    
    #训练或者测试集索引
    split_lab_filenames = sorted([file for file in os.listdir(opt.annotation_path) if file.strip('.txt')[-1] ==str(split)])
       
    data = []
    for file in split_lab_filenames:
        class_id = '_'.join(os.path.splitext(file)[0].split('_')[:-2])
        f = open(os.path.join(opt.annotation_path, file), 'r')
        for line in f: 
            if train==1 and line.split(' ')[1] == '1':#训练数据集数据名字及标签读取
                frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                if os.path.exists(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))
            elif train==2 and line.split(' ')[1] == '2':#验证数据集数据名字及标签读取
                frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                if os.path.exists(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))
            elif train==0 and line.split(' ')[1] == '0':#测试数据集数据名字及标签读取
                frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                if os.path.exists(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))
  
        f.close()

    #用于给动态网络提供处理后的数据
    def inner():
        for idx in range(len(data)):
            video = data[idx]
            label_id = lab_names.get(video[1])
            frame_path = os.path.join(opt.frame_dir, video[1], video[0])
            if opt.only_RGB:#图像数目
                Total_frames = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))
            else:#光流图像数目
                Total_frames = len(glob.glob(glob.escape(frame_path) +  '/TVL1jpg_y_*.jpg'))

            if train_val_test == 0: 
                clip = get_test_video(opt, frame_path, Total_frames)
            else:
                clip = get_train_video(opt, frame_path, Total_frames)
            yield scale_crop(clip, train_val_test, opt), int(label_id)
    return inner
