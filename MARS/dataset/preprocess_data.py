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
from PIL import Image, ImageFilter, ImageOps, ImageChops
import numpy as np
import random
import numbers
import pdb
import time

try:
    import accimage
except ImportError:
    accimage = None
    
scale_choice = [1, 1/2**0.25, 1/2**0.5, 1/2**0.75, 0.5]
crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

def PIL2numpy(pic,norm_value=255.0):
    """
    将PIL.Image图像转化为numpy张量，并对图像像素值进行归一化
    参数:
        pic (PIL.Image or numpy.ndarray): 要转化为张量的PIL.Image图像.
        norm_value：归一化因子
    Returns:
        Tensor: 转化后的numpy张量.
    """
    #如果pic本身就是numpy张量
    if isinstance(pic, np.ndarray):
        #将WxHxC的图像转化为CxWxH
        img = pic.transpose((2, 0, 1))
        return img/norm_value
    
    #如果pic是accimage.Image类型的数据
    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return nppic

    #处理PIL Image
    if pic.mode == 'I':
        img = np.array(pic, np.int32, copy=False)
    elif pic.mode == 'I;16':
        img = np.array(pic, np.int16, copy=False)
    else:
        img = np.frombuffer(pic.tobytes(),dtype="uint8")
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.reshape(pic.size[1], pic.size[0], nchannel)
    img = np.array(np.transpose(img,axes=(2,0,1)),dtype="float32")
    return img

def Scale(img, size, interpolation=Image.BILINEAR):
    """将给定的PIL.Image图像缩放到给定大小
    参数:
        img: 输入的PIL.Image图像
        size (sequence or int): 理想的输出图像大小
        interpolation (int, optional): 图像缩放使用的插值模式
    Returns:
        PIL.Image: 缩放后的图像
    """
    if isinstance(size, int):#如果size是一个整数，将图像的最长边缩放到给定大小
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size, interpolation)
    
def CenterCrop(img, size):
    """
    对给定的PIL.Image进行中心裁剪
    参数:
        img: 输入的PIL.Image图像
        size (sequence or int): 裁剪后理想的输出图像大小
    Return:
         PIL.Image: 裁剪后的图像
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    w, h = img.size
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img.crop((x1, y1, x1 + tw, y1 + th))

def MultiScaleCornerCrop(img, scale, size, crop_position, interpolation=Image.BILINEAR):
    """
    将给定的PIL.Image裁剪为随机选择的大小。
    从原始尺寸的比例中选择尺寸的裁剪。
    裁剪的位置是从4个角和1个中心中随机选择的。
    最终将裁剪后的图像调整为给定大小。
    参数:
        img: 输入的PIL.Image图像
        scales: 原始大小的裁剪尺度
        size: 短边的大小
        interpolation: 图像缩放使用的插值模式
    Return:
        PIL.Image:裁剪并调整大小后的图像
    """  
    min_length = min(img.size[0], img.size[1])
    crop_size = int(min_length * scale)

    image_width = img.size[0]
    image_height = img.size[1]
    if crop_position == 'c': #中心裁剪
        center_x = image_width // 2
        center_y = image_height // 2
        box_half = crop_size // 2
        x1 = center_x - box_half
        y1 = center_y - box_half
        x2 = center_x + box_half
        y2 = center_y + box_half
    elif crop_position == 'tl':#左上角裁剪
        x1 = 0
        y1 = 0
        x2 = crop_size
        y2 = crop_size
    elif crop_position == 'tr':#右上角裁剪
        x1 = image_width - crop_size
        y1 = 0
        x2 = image_width
        y2 = crop_size
    elif crop_position == 'bl':#左下角裁剪
        x1 = 0
        y1 = image_height - crop_size
        x2 = crop_size
        y2 = image_height
    elif crop_position == 'br':#右下角裁剪
        x1 = image_width - crop_size
        y1 = image_height - crop_size
        x2 = image_width
        y2 = image_height
    img = img.crop((x1, y1, x2, y2))
    return img.resize((size, size), interpolation)

def Normalize(img, mean, std):
    """
    用均值和标准差对张量图像进行归一化。
    参数:
        img: 输入的PIL.Image图像
        mean (sequence): 分别用于R，G，B通道的均值序列
        std (sequence):  分别用于R，G，B通道的标准差序列
        Returns:
            Tensor: 归一化后的图像
    """
    for id, m, s in zip(range(len(mean)),mean, std):
        img[id,:,:]=(img[id,:,:]-m)/s
    return img

def RandomHorizontalFlip(img,p):
    """
    水平翻转给定的PIL.Image，概率为0.5。
    参数:
        img (PIL.Image): 要被翻转的图像.
    Returns:
        PIL.Image: 随机翻转后的图像.
    """
    if p < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
        
def get_mean( dataset='HMDB51'):
    #获取数据集的均值
    if dataset == 'activitynet':
        return [114.7748, 107.7354, 99.4750 ]
    elif dataset == 'kinetics':
        return [110.63666788, 103.16065604,  96.29023126]
    elif dataset == "HMDB51":
        return [0.36410178082273*255, 0.36032826208483*255, 0.31140866484224*255]

def get_std(dataset = 'HMDB51'):
    #获取数据集的标准差
    if dataset == 'kinetics':
        return [38.7568578, 37.88248729, 40.02898126]
    elif dataset == 'HMDB51':
        return [0.20658244577568*255, 0.20174469333003*255, 0.19790770088352*255]

def scale_crop(clip, train, opt): 
    """
    预处理训练/测试视频图像或光流
    训练:
        - 多尺度随机裁剪
        - 随机水平翻转
        - 将PIL图像转化为numpy张量
        - 使用``ActivityNet``的均值和标准差对R、G、B进行归一化
    测试或验证:
        - 缩放帧
        - 中心裁剪
        - 将PIL图像转化为numpy张量
        - 使用``ActivityNet``的均值和标准差对R、G、B进行归一化
    参数:
        clip (list(frames)): RGB或者光流帧列表
        train : 1为训练, 0为测试
    Return:
        Tensor(frames) of shape C x T x H x W
    """
    if opt.modality == 'RGB':
        processed_clip = np.zeros(shape=(3, len(clip), opt.sample_size, opt.sample_size))
    elif opt.modality == 'Flow':
        processed_clip = np.zeros(shape=(2, int(len(clip)/2), opt.sample_size, opt.sample_size))
    elif opt.modality == 'RGB_Flow':
        processed_clip = np.zeros(shape=(5, int(len(clip)/3), opt.sample_size, opt.sample_size))
    
    flip_prob     = random.random()
    scale_factor  = scale_choice[random.randint(0, len(scale_choice) - 1)]
    crop_position = crop_positions[random.randint(0, len(crop_positions) - 1)] 
    #对于训练
    if train == 1:
        j = 0
        for i, I in enumerate(clip):
            #对图像进行随机
            I = MultiScaleCornerCrop(I,scale = scale_factor, size = opt.sample_size, crop_position = crop_position)
            #对图像进行随机水平翻转
            I = RandomHorizontalFlip(I,p = flip_prob)
            #只使用RGB图像时
            if opt.modality == 'RGB':
                #将RGB图像转化为numpy张量
                I = PIL2numpy(I,1.0)
                #对RGB图像进行归一化
                I = Normalize(I, get_mean('activitynet'),[1,1,1])
                processed_clip[:, i, :, :] = I
            #只使用光流图像时
            elif opt.modality == 'Flow':
                if i%2 == 0 and flip_prob<0.5:
                    #对光流图像进行水平翻转
                    I = ImageChops.invert(I)
                #将光流图像转化为numpy张量
                I = PIL2numpy(I,1.0)
                #对光流图像进行归一化
                I = Normalize(I,[127.5], [1])
                if i%2 == 0:
                    processed_clip[0, int(i/2), :, :] = I
                elif i%2 == 1:
                    processed_clip[1, int((i-1)/2), :, :] = I
            #使用RGB图像和光流图像时
            elif opt.modality == 'RGB_Flow':
                if j == 1 and flip_prob<0.5:
                    #对光流图像进行水平翻转
                    I = ImageChops.invert(I)
                #将图像转化为numpy张量
                I = PIL2numpy(I,1.0)                          
                if j == 0:
                    #对RGB图像进行归一化
                    I = Normalize(I, get_mean('activitynet'), [1,1,1,])
                    processed_clip[0:3, int(i/3), :, :] = I
                else:
                    #对光流图像进行归一化
                    I = Normalize(I, [127.5], [1])
                    if j == 1:
                        processed_clip[3, int((i-1)/3), :, :] = I
                    elif j == 2:
                        processed_clip[4, int((i-2)/3), :, :] = I
                j += 1
                if j == 3:
                    j = 0
    #对于测试和验证
    else:
        j = 0
        for i, I in enumerate(clip):
            #对图像进行缩放
            I = Scale(I, opt.sample_size)
            #对图像进行中心裁剪
            I = CenterCrop(I, opt.sample_size)
            #将PIL.Image图像转化为numpy张量
            I = PIL2numpy(I,1.0)
            if opt.modality == 'RGB':#只是用RGB图像
                I = Normalize(I, get_mean('activitynet'), [1,1,1])
                processed_clip[:, i, :, :] = I
            elif opt.modality == 'Flow':#只是用光流图像
                I = Normalize(I,[127.5], [1])
                if i%2 == 0:
                    processed_clip[0, int(i/2), :, :] = I
                elif i%2 == 1:
                    processed_clip[1, int((i-1)/2), :, :] = I
            elif opt.modality == 'RGB_Flow':#只是用RGB和光流图像
                if j == 0:
                    I = Normalize(I,get_mean('activitynet'), [1,1,1,])
                    processed_clip[0:3, int(i/3), :, :] = I
                else:
                    I = Normalize(I,[127.5], [1])                  
                    if j == 1:
                        processed_clip[3, int((i-1)/3), :, :] = I
                    elif j == 2:
                        processed_clip[4, int((i-2)/3), :, :] = I
                j += 1
                if j == 3:
                    j = 0
                    
    return(processed_clip)