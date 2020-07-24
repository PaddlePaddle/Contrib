#coding=utf-8
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
    if Total_frames < opt.sample_duration: loop = 1
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
                if opt.only_RGB and os.path.exists(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))
                elif os.path.exists(frame_path) and "done" in os.listdir(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))
            elif train!=1 and line.split(' ')[1] == '2':#验证或者测试数据集数据名字及标签读取
                frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                if opt.only_RGB and os.path.exists(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))
                elif os.path.exists(frame_path) and "done" in os.listdir(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))
  
        f.close()

    #用于给动态网络提供处理后的数据
    def inner():
        #将数据进行打乱
        np.random.shuffle(data)
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

def UCF101_test(train, opt, split=None):
    """UCF101 Dataset"""
    """
    Args:
        opt   : config options
        train : 0 for testing, 1 for training, 2 for validation 
        split : 1,2,3 
    Returns:
        (tensor(frames), class_id ): Shape of tensor C x T x H x W
    """
    train_val_test = train
    with open(os.path.join(opt.annotation_path, "classInd.txt")) as lab_file:
        lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]
        
    with open(os.path.join(opt.annotation_path, "classInd.txt")) as lab_file:
        index = [int(line.strip('\n').split(' ')[0]) for line in lab_file]

    # Number of classes
    class_num = len(lab_names)
    assert class_num == 101
    class_idx = dict(zip(lab_names, index))   # Each label is mappped to a number
    idx_class = dict(zip(index, lab_names))   # Each number is mappped to a label
    
    # indexes for training/test set
    split_lab_filenames = sorted([file for file in os.listdir(opt.annotation_path) if file.strip('.txt')[-1] ==str(split)])
    if train_val_test==1:
        split_lab_filenames = [f for f in split_lab_filenames if 'train' in f]
    else:
        split_lab_filenames = [f for f in split_lab_filenames if 'test' in f]
    data = []                                     # (filename , lab_id)
        
    f = open(os.path.join(opt.annotation_path, split_lab_filenames[0]), 'r')
    for line in f:
        class_id = class_idx.get(line.split('/')[0]) - 1
        if os.path.exists(os.path.join(opt.frame_dir, line.strip('\n')[:-4])) == True:
            data.append((os.path.join(opt.frame_dir, line.strip('\n')[:-4]), class_id))
        
    f.close()
    def inner():
        np.random.shuffle(data)
        for idx in range(len(data)):
            video = data[idx]
            label_id = lab_names.get(video[1])
            frame_path = os.path.join(opt.frame_dir, idx_class.get(label_id + 1), video[0])
            if opt.only_RGB:
                Total_frames = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))
            else:
                Total_frames = len(glob.glob(glob.escape(frame_path) +  '/TVL1jpg_y_*.jpg'))

            if train_val_test == 0: 
                clip = get_test_video(opt, frame_path, Total_frames)
            else:
                clip = get_train_video(opt, frame_path, Total_frames)

            yield scale_crop(clip, train_val_test, opt), int(label_id)
    return inner


def Kinetics_test(train, opt, split=None):
    """
    Args:
        opt   : config options
        train : 0 for testing, 1 for training, 2 for validation 
        split : 'val' or 'train'
    Returns:
        (tensor(frames), class_id ) : Shape of tensor C x T x H x W
    """
    train_val_test = train
              
    # joing labnames with underscores
    lab_names = sorted([f for f in os.listdir(os.path.join(opt.frame_dir, "train"))])        
       
    # Number of classes
    label_num = len(lab_names)
    assert label_num == 400
        
    # indexes for validation set
    if train==1:
        label_file = os.path.join(opt.annotation_path, 'Kinetics_train_labels.txt')
    else:
        label_file = os.path.join(opt.annotation_path, 'Kinetics_val_labels.txt')

    data = []                                     # (filename , lab_id)
    
    f = open(label_file, 'r')
    for line in f:
        class_id = int(line.strip('\n').split(' ')[-2])
        nb_frames = int(line.strip('\n').split(' ')[-1])
        data.append((os.path.join(opt.frame_dir,' '.join(line.strip('\n').split(' ')[:-2])), class_id, nb_frames))
    f.close()
            

    def inner():
        np.random.shuffle(data)
        for idx in range(len(data)):
            video = data[idx]
            label_id = video[1]
            frame_path = video[0]
            Total_frames = video[2]
    
            if opt.only_RGB:
                Total_frames = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))  
            else:
                Total_frames = len(glob.glob(glob.escape(frame_path) +  '/TVL1jpg_y_*.jpg'))
    
            if train_val_test == 0: 
                clip = get_test_video(self.opt, frame_path, Total_frames)
            else:
                clip = get_train_video(self.opt, frame_path, Total_frames)
            yield scale_crop(clip, train_val_test, opt), int(label_id)
    return inner

    

