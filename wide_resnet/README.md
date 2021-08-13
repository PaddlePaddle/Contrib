# Wide_Resnet

[简体中文](./README_ch.md) | English
   
   * [Wide Resnet](#Wide_Resnet)
      * [1_Introduction](#1_Introduction)
      * [2_Accuracy](#2_Accuracy)
      * [3_Dataset](#3_Dataset)
      * [4_Environment](#4_Environment)
      * [5_Quick_start](#5_Quick_start)
         * [step1: clone](#step1-clone)
         * [step1: train](#step2-train)
         * [step2: eval](#step3-eval)
      * [6_Code_structure](#6_Code_structure)
         * [6.1_structure](#61-structure)
         * [6.2 _Parameter_description](#62_Parameter_description)
         * [6.3_train_process](#63_train_process)
            * [Single_machine_training](#Single_machine_training)
            * [Multi_machine_training](#Multi_machine_training)
            * [Training_output](#Training_output)
         * [6.4_Evaluation_process](#64-Evaluation_process)
      * [7_Model_information](#7_Model_information)

## 1_Introduction
This project reproduces Wide Resnet based on the paddlepaddle framework. It is a variation of ResNet. The main difference lies in the improvement of shortcut of ResNet, the use of "wider" convolution and the addition of dropout layer.


**paper:**
- [1]  Zagoruyko S ,  Komodakis N . Wide Residual Networks[J].  2016.<br>
- link：[Wide Residual Networks](https://arxiv.org/abs/1605.07146)

**Reference items:**
- [https://github.com/xternalz/WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch)
- [https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py](https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py)

**Link of aistudio:**
[https://aistudio.baidu.com/aistudio/projectdetail/2251959](https://aistudio.baidu.com/aistudio/projectdetail/2251959)

## 2_Accuracy

>The indicators are tested in the test set of cifar10

train from scratch:


| |epoch|opt|batch_size|dataset|memory|card|precision|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|1|400|SGD|128|CIFAR10|16G|1|0.9660|

**model download**
link：[aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104172)


## 3_Dataset

[CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html)

- Dataset size：
  - train set:50000
  - eval set:10000
  - image size:32 * 32
- Data format：Classification dataset

## 4_Environment

- Hardware：GPU CPU

- Frame:
  - PaddlePaddle >= 2.0.0

# 5_Quick_start

### step1-clone

```bash
# clone this repo
git clone https://github.com/PaddlePaddle/Contrib.git
cd wide_resnet
export PYTHONPATH=./
```
**install**
```bash
python3 -m pip install -r requirements.txt
```

### step2-train
```bash
python3 train.py
```
If you want distributed training and use multiple cards:
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py
```

output：
```
Epoch 0: PiecewiseDecay set learning rate to 0.05.
iter:0  loss:2.4832
iter:10  loss:2.3544
iter:20  loss:2.3087
iter:30  loss:2.2509
iter:40  loss:2.2450
```

### step3-eval
```bash
python3 eval.py
```
output：
```
acc:9660 total:10000 ratio:0.966
```

## 6_Code_structure

### 61-structure

```
│  wide_resnet.py                 # model file
│  eval.py                        # eval script
│  README.md                      # english readme
│  README_cn.md                   # chinese readme
│  requirements.txt                # requirement
│  train.py                       # train script
```

### 62_Parameter_description

None


### 63_train_process

#### Single_machine_training
```bash
python3 train.py
```

#### Multi_machine_training
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py
```

At this time, the program will import the output log of each process to the debug path:
```
.
├── debug
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
├── README.md
└── train.py
```

#### Training_output
After performing the training, you will get an output similar to the following. Each round of 'batch' training will print the current epoch, step and loss values.
```text
Epoch 0: PiecewiseDecay set learning rate to 0.05.
iter:0  loss:2.4832
iter:10  loss:2.3544
iter:20  loss:2.3087
iter:30  loss:2.2509
iter:40  loss:2.2450
```

### 64-Evaluation_process

```bash
python3 eval.py
```

output:
```
acc:9660 total:10000 ratio:0.966
```
## 7_Model_information

For other information about the model, refer to the following table:

| Anformation | Description |
| --- | --- |
| Author | mingyuan xu|
| Time | 2021.08 |
| Frame Version | >=Paddle 2.0.2|
| Application scenario | Image classification |
| Hardware | GPU、CPU |
| download Link |[aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104172) |
| Online Link | [botebook](https://aistudio.baidu.com/aistudio/projectdetail/2251959)|
