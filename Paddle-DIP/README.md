# Deep image prior

[English](./README_EN.md) | 简体中文
   
   * [DIP](#resnet)
      * [一、简介](#一简介)
      * [二、复现精度](#二复现精度)
      * [三、数据集](#三数据集)
      * [四、环境依赖](#四安装环境依赖)
      * [五、快速开始](#五快速开始)
      * [六、代码结构与详细说明](#六代码结构与详细说明)
         * [6.1 代码结构](#61-代码结构)
         * [6.2 训练流程](#62-训练流程)
      * [七、模型信息](#七模型信息)

## 一、简介

本项目基于 paddlepaddle 框架复现 Deep image prior。Deep image prior 使用神经网络进行图像重建但无需学习。


**论文名称**： [Deep Image Prior](https://arxiv.org/pdf/1711.10925.pdf)

**数据集**： Set14 数据集： [https://github.com/jbhuang0604/SelfExSR/tree/master/data](https://github.com/jbhuang0604/SelfExSR/tree/master/data)

**验收标准**： 8 × super-resolution, avg psnr = 24.15%

**参考项目：** [https://github.com/DmitryUlyanov/deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior)

**项目aistudio地址**：https://aistudio.baidu.com/aistudio/projectdetail/2251959

## 二、复现精度

> 该列指标在 Set14 数据集上的测试结果

|num_inter | optimizer | dataset | memory | card | avg psnr |
| --- | --- | --- | --- | --- | --- |
| 4000 | Adam | Set 14 | 16G | 1 | 24.1653 |

## 三、数据集

Set14 数据集： [https://github.com/jbhuang0604/SelfExSR/tree/master/data](https://github.com/jbhuang0604/SelfExSR/tree/master/data)

- 数据集概要：
  - 原图：14 张
  - 图片名称：
    `["baboon", "barbara", "bridge", "coastguard", "comic", "face", "flowers", "foreman", "lenna", "man", "monarch", "pepper", "ppt3", "zebra"]`

## 四、安装环境依赖

- 硬件：GPU、CUDA、cuDNN、CPU

- 框架：
  - PaddlePaddle >= 2.0.0
  - matplotlib = 3.4
  - sikt-image = 0.18

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/PaddlePaddle/Contrib.git
cd Paddle-DIP
```

### step2: 打开 `super-resolution.ipynb`
首先，在 **Load data and optimize** 中指定你本地的 Set 14 数据集路径。

我们运行的输出结果也可直接在 `super-resolution.ipynb` 中浏览，包括：

- 输出结果图像栅格
  ![1](https://img-blog.csdnimg.cn/fca0b10c09154e87bd4e15d59ce78176.png)
- 中间输出：
	```
	...
	baboon: PSNR_HR = 21.438
	HR and LR resolutions: (704, 576), (88, 72)
	...
	```
- 最终精度
	```
	24.16529286849991
	```


## 六、代码结构与详细说明

### 6.1 代码结构

```
├─datas                           # 存放 Set14 数据集
├─models                          # 模型
├─utils                           # 工具代码
├─super-resolution.ipynb          # 训练+可视化 notebook
├─README.md                       # 中文 readme
├─README_EN.md                    # english readme
```

### 6.2 训练流程
打开并运行 `super-resolution.ipynb`

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | 钱坤、石华榜、|
| 时间 | 2021.08 |
| 框架版本 | >= Paddle 2.1.2|
| 应用场景 | 图像重建 |
| 支持硬件 | GPU、CPU |
| 下载链接 |[aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104172) |

