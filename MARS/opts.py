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

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    ###############################
    #       数据集相关的设置
    ###############################
    #图像位置
    parser.add_argument('--frame_dir',default='dataset/HMDB51/',type=str,help='path of jpg files')
    #标签位置
    parser.add_argument('--annotation_path',default='dataset/HMDB51_labels',type=str,help='label paths')
    #使用哪个数据集
    parser.add_argument('--dataset',default='HMDB51',type=str,help='(HMDB51, UCF101, Kinectics)')
    #数据划分
    parser.add_argument('--split',default=1,type=str,help='(for HMDB51 and UCF101)')
    #使用的图片的形态，RGB？FLOW？
    parser.add_argument('--modality',default='RGB',type=str,help='(RGB, Flow)')
    #输入的通道数目
    parser.add_argument('--input_channels',default=3,type=int,help='(3, 2)')
    #从零开始训练时，数据集总的类别数
    parser.add_argument('--n_classes',default=51,type=int,help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    #从预训练模型微调时，数据集总的类别数
    parser.add_argument('--n_finetune_classes',default=51,type=int,help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    #仅提取RGB帧
    parser.add_argument('--only_RGB', action='store_true', help='Extracted only RGB frames')
    parser.set_defaults(only_RGB = False)
    ###############################
    #     模型相关的参数设置
    ###############################
    #前向传播过程中的输出层
    parser.add_argument('--output_layers',action='append',help='layer to output on forward pass')
    parser.set_defaults(output_layers=[])
    #模型的基本骨干网络
    parser.add_argument('--model',default='resnext',type=str,help='Model base architecture')
    #模型深度
    parser.add_argument('--model_depth',default=101,type=int,help='Number of layers in model')
    #Resnet的截断方式
    parser.add_argument('--resnet_shortcut',default='B',type=str,help='Shortcut type of resnet (A | B)')
    #
    parser.add_argument('--resnext_cardinality', default=32,type=int,help='ResNeXt cardinality')
    #微调时从哪个块开始调整
    parser.add_argument('--ft_begin_index',default=4,type=int,help='Begin block index of fine-tuning')
    #输入数据的长宽
    parser.add_argument('--sample_size',default=112,type=int,help='Height and width of inputs')
    #输入暂存的时间长度
    parser.add_argument('--sample_duration',default=16,type=int,help='Temporal duration of inputs')
    #哪个阶段，训练还是测试？
    parser.add_argument('--training', action='store_true', help='training/testing')
    parser.set_defaults(training=True)
    parser.add_argument('--freeze_BN', action='store_true', help='freeze_BN/testing')
    parser.set_defaults(freeze_BN=False)
    #训练时的批大小
    parser.add_argument('--batch_size', default=20, type=int, help='Batch Size')
    #数据加载时的worker数目  
    parser.add_argument('--n_workers', default=4, type=int, help='Number of workers for dataloader')

    # optimizer parameters
    ###############################
    #     参数优化的参数设置
    ###############################
    #学习率
    parser.add_argument('--learning_rate',default=0.1,type=float,help='Initial learning rate (divided by 10 while training by lr scheduler)')
    #动量
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    #权重衰减因子
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    #
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    #优化器
    parser.add_argument('--optimizer',default='sgd',type=str,help='Currently only support SGD')
    #
    parser.add_argument('--lr_patience',default=10,type=int,help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    #光流权重增加了MSE损失
    parser.add_argument('--MARS_alpha', default=50, type=float, help='Weight of Flow augemented MSE loss')
    #总共训练多少个迭代
    parser.add_argument('--n_epochs',default=100,type=int,help='Number of total epochs to run')
    
    parser.add_argument('--begin_epoch',default=1,type=int,help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    ###############################
    #     日志的参数设置
    ###############################
    #结果保存的路径
    parser.add_argument('--result_path',default='',type=str,help='result_path')
    #测试MARS
    parser.add_argument('--MARS', action='store_true', help='test MARS')
    parser.set_defaults(MARS=False) 
    #预训练模型的训练
    parser.add_argument( '--MARS_premodel_path', default='', type=str, help='MARS pretrain model')
    parser.add_argument( '--Flow_premodel_path', default='', type=str, help='FLOW pretrain model')
    parser.add_argument( '--RGB_premodel_path', default='', type=str, help='RGB pretrain model')
    #MARS模型保存路径
    parser.add_argument( '--MARS_resume_path', default='', type=str, help='MARS resume model')
    parser.add_argument( '--Flow_resume_path', default='', type=str, help='FLOW resume model')
    parser.add_argument( '--RGB_resume_path', default='', type=str, help='RGB resume model')
    parser.add_argument( '--continue_train', default=False, type=bool, help='')
    #训练模型时，每迭代多少轮，保存一次
    parser.add_argument('--checkpoint',default=2,type=int,help='Trained model is saved at every this epochs.')
    #手动设置随机种子
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    #手动设置抽样验证片段的随机种子
    parser.add_argument('--random_seed', default=1, type=bool, help='Manually set random seed of sampling validation clip')
    args = parser.parse_args()
    return args
