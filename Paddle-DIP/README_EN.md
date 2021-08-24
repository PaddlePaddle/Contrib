# Deep image prior

English | [简体中文](./README.md)
   
   * [DIP](#resnet)
      * [1、Introduction](#1-Introduction)
      * [2、Accuracy](#2-Accuracy)
      * [3、Dataset](#3-Dataset)
      * [4、Environment](#4-Environment)
      * [5、Quick start](#5-Quick-start)
      * [6、Code structure](#6-Code-structure)
         * [6.1 structure](#61-structure)
         * [6.2 Train process](#62-Train-process)
      * [7、Model info](#7-Model-info)

## 1 Introduction

This project is based on the paddlepaddle framework to reproduce Deep image prior. Deep image prior uses neural networks for image reconstruction but does not require learning.


**Paper name**： [Deep Image Prior](https://arxiv.org/pdf/1711.10925.pdf)

**dataset**： Set14 数据集： [https://github.com/jbhuang0604/SelfExSR/tree/master/data](https://github.com/jbhuang0604/SelfExSR/tree/master/data)

**Acceptance criteria**： 8 × super-resolution, avg psnr = 24.15%

**Reference project：** [https://github.com/DmitryUlyanov/deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior)

**AI studio adress**：https://aistudio.baidu.com/aistudio/projectdetail/2266236

## 2 Accuracy

> Test results of this column of indicators on set14 dataset

|num_inter | optimizer | dataset | memory | card | avg psnr |
| --- | --- | --- | --- | --- | --- |
| 4000 | Adam | Set 14 | 16G | 1 | 24.1653 |

## 3 Dataset

Set14 dataset： [https://github.com/jbhuang0604/SelfExSR/tree/master/data](https://github.com/jbhuang0604/SelfExSR/tree/master/data)

- Dataset summary：
  - raw images：14 pictures
  - images name：
    `["baboon", "barbara", "bridge", "coastguard", "comic", "face", "flowers", "foreman", "lenna", "man", "monarch", "pepper", "ppt3", "zebra"]`

## 4 Environment

- Hardware：GPU、CUDA、cuDNN、CPU

- Framework：
  - PaddlePaddle >= 2.0.0
  - matplotlib = 3.4
  - sikt-image = 0.18
  - opencv-python = 4.5.3

## 5 Quick start

### step1: clone 

```bash
# clone this repo
git clone https://github.com/PaddlePaddle/Contrib.git
cd Paddle-DIP
```

### step2: Open `super-resolution.ipynb`

First, specify your local Set14 dataset path in **Load data and optimize**.

The output of our operation can also be browsed directly in `super-resolution.ipynb`, including:

- Output result image grid
  ![1](https://img-blog.csdnimg.cn/fca0b10c09154e87bd4e15d59ce78176.png)
- Intermediate output：
	```
	...
	baboon: PSNR_HR = 21.438
	HR and LR resolutions: (704, 576), (88, 72)
	...
	```
- Final accuracy
	```
	24.16529286849991
	```


## 6 Code structure

### 6.1 Structure

```
├─datas                           # Storing set14 datasets
├─models                          # Neural network model
├─utils                           # Tool code
├─super-resolution.ipynb          # Training + visualization notebook
├─README.md                       # 中文 readme
├─README_EN.md                    # english readme
```

### 6.2 Train process
Opening an run `super-resolution.ipynb`

## 7 Model info

For other information about the model, refer to the following table：

| information | directions |
| --- | --- |
| publisher | [KunStats](https://github.com/KunStats)、李芯瑶、[S-HuaBomb](https://github.com/S-HuaBomb)、|
| time | 2021.08 |
| Framework version | >= Paddle 2.1.2|
| Application scenario | image reconstruction  |
| Support hardware | GPU、CPU |
| AI Studio link |[aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/2266236) |

