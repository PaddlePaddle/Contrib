# PSENet

English | [简体中文](./README_cn.md)
   
   * [PSENet](#resnet)
      * [1 Introduction](#1 Introduction)
      * [2 Accuracy](#2 Accuracy)
      * [3 Dataset](#3 Dataset)
      * [4 Environment](#4 Environment)
      * [5 Quick start](#5 Quick start)
         * [step1: train](#step1-train)
         * [step2: eval](#step2-eval)
         * [step3: test](#step3-test)
      * [6 Code structure](#6 Code structure)
         * [6.1 structure](#61-structure)
         * [6.2 Parameter description](#62-Parameter description)
         * [6.3 Training process](#63-Training process)
            * [Single machine training](#Single machine training)
            * [Multi machine training](#Multi machine training)
            * [Training output](#Training output)
         * [6.4 Evaluation process](#64-Evaluation process)
         * [6.5 Test process](#65-Test process)
         * [6.6 Prediction using pre training model](#66-Prediction using pre training model)
      * [7 Model information](#7 Model information)

## 1 Introduction

This project reproduces PSENet based on paddlepaddle framework. PSENet is a new instance segmentation network. It has two advantages. Firstly, as a segmentation based method, psenet can locate arbitrary shape text. Secondly, the model proposes a progressive scaling algorithm, which can successfully recognize adjacent text instances.


**Paper:**
- [1] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao. Shape robust text detection with progressive scale expansion network. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 9336–9345, 2019.<br>

**Reference project：**
- [https://github.com/whai362/PSENet](https://github.com/whai362/PSENet)

**The link of aistudio：**
- notebook：[https://aistudio.baidu.com/aistudio/projectdetail/1945560](https://aistudio.baidu.com/aistudio/projectdetail/1945560)
- Script：[https://aistudio.baidu.com/aistudio/clusterprojectdetail/1796445](https://aistudio.baidu.com/aistudio/clusterprojectdetail/1796445)

## 2 Accuracy

>This index is tested in the test set of icdar2015

train from scratch：


| |epoch|opt|short_size|batch_size|dataset|memory|card|precision|recall|hmean|FPS|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|pretrain_1|33|Adam|1024|16|ICDAR2017|32G|1|0.68290|0.68850|0.68569|5.0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_2|46|Adam|1024|16|ICDAR2013、ICDAR2017、COCO_TEXT|32G|4|0.69678|0.69812|0.69745|5.0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_3|68|Adam|1260|16|ICDAR2013、ICDAR2015、ICDAR2017、COCO_TEXT|32G|1|0.86526|0.80693|0.83508|2.0|[psenet_r50_ic17_1260.py](./config/psenet/psenet_r50_ic17_1260.py)|


**ICDAR2015**
>This index is tested in the test set of icdar2015

Training details:

| |pretrain|epoch|opt|short_size|batch_size|dataset|memory|card|precision|recall|hmean|FPS|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|finetune_1|pretrain_1|491|Adam|1024|16|ICDAR2015|32G|1|0.86463|0.80260|0.83246|5.0|[finetune1.py](./config/psenet/finetune1.py)|
|finetune_2|pretrain_3|-|Adam|1260|16|ICDAR2015|32G|1|0.87024|0.81367|0.84101|2.0|[finetune2.py](./config/psenet/finetune2.py)|
|finetune_3|finetune_2|401|SGD|1480|16|ICDAR2015|32G|1|<font color='red'>0.88060</font>|<font color='red'>0.82378</font>|<font color='red'>0.85124</font>|<font color='red'>1.8</font>|[finetune3.py](./config/psenet/finetune3.py)|

**Total_text**
>This column of indicators is in total_ Test set test of text

Training details:

| |pretrain|epoch|opt|short_size|batch_size|dataset|memory|card|precision|recall|hmean|FPS|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|finetune_1|None|331|Adam|736|16|Total_Text|32G|1|0.84823|0.76007|0.80173|10.1|[psenet_r50_tt.py](./config/psenet/psenet_r50_tt.py)|
|finetune_2|pretrain_2|290|Adam|736|16|Total_Text|32G|1|<font color='red'>0.88482</font>|<font color='red'>0.79002</font>|<font color='red'>0.83474</font>|<font color='red'>10.1</font>|[psenet_r50_tt_finetune2.py](./config/psenet/psenet_r50_tt_finetune2.py)|

**Model Download**
Address：[Google cloud](https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing)

Detailed information：

|name|path|configuration file|
| :---: | :---: | :---: |
|pretrain_1|psenet_r50_ic17_1024_Adam/checkpoint_33_0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_2|psenet_r50_ic17_1024_Adam/checkpoint_46_0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_3|psenet_r50_ic17_1260_Adam/checkpoint_68_0|[psenet_r50_ic17_1260.py](./config/psenet/psenet_r50_ic17_1260.py)|
|ic15_finetune_1|psenet_r50_ic15_1024_Adam/checkpoint_491_0|[finetune1.py](./config/psenet/finetune1.py)|
|ic15_finetune_2|psenet_r50_ic15_1260_Adam/best|[finetune2.py](./config/psenet/finetune2.py)|
|ic15_finetune_3|psenet_r50_ic15_1480_SGD/checkpoint_401_0|[finetune3.py](./config/psenet/finetune3.py)|
|tt_finetune_1|psenet_r50_tt/checkpoint_331_0|[psenet_r50_tt.py](./config/psenet/psenet_r50_tt.py)|
|tt_finetune_2|psenet_r50_tt/checkpoint_290_0|[psenet_r50_tt_finetune2.py](./config/psenet/psenet_r50_tt_finetune2.py)|

## 3 Dataset

[Icdar2015 text detection dataset](https://rrc.cvc.uab.es/?ch=4&com=downloads)。

- Dataset size：
  - train：1000
  - test：500
- Data format：Rectangle text dataset

[Total text detection dataset](https://github.com/cs-chan/Total-Text-Dataset)。


- Dataset size：
  - train：1255
  - test：300
- Data format: Curved text dataset

## 4 Environment

- Hardware: GPU, CPU

- Framework:
  - PaddlePaddle >= 2.0.0

## 5 Quick start

### step1: clone 

```bash
# clone this repo
git clone https://github.com/PaddlePaddle/Contrib.git
cd PSENet
export PYTHONPATH=./
```
**Installation dependency**
```bash
sh init.sh
```

**Compile PSE**
Because author provided the PSP code of CPP to speed up the post-processing process, we need to compile it. Note that we have already packaged the compiled code in aistudio, so we don't need to perform this step!
```bash
sh compile.sh
```
If the compilation is not successful, you can use the provided Python version of PSE Code:
[pypse.py](./models/pypse.py)


### step2: train
```bash
python3 train.py ./config/psenet/psenet_r50_ic15_1024_Adam.py
```
If the training is interrupted, it can be recovered through the -- resume parameter, for example, using the -- resume checkpoint_ 44_ 0 means the interrupt is resumed at 44epoch and 0 iter:
```bash
python3 train.py ./config/psenet/psenet_r50_ic15_1024_Adam.py --resume checkpoint_44_0 # Is not the absolute path of the breakpoint parameter, please note
```
If you want to train distributed and use multicards:
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py config/psenet/psenet_r50_ic17_1024_Adam.py
```

The output is:
```
Epoch: [1 | 600]
(1/78) LR: 0.001000 | Batch: 6.458s | Total: 0min | ETA: 8min | Loss: 0.614 | Loss(text/kernel): 0.506/0.109 | IoU(text/kernel): 0.274/0.317 | Acc rec: 0.000
```
Because it is a target detection task, we need to pay attention to the gradual decrease of ``loss`` and the gradual increase of ``IOU``.

### step3: test
```bash
python3 eval.py ./config/psenet/psenet_r50_ic15_1024_Adam.py ./checkpoints/psenet_r50_ic15_1024_Adam/checkpoint_491_0.pdparams --report_speed
```
The output is:
```
Testing 1/500
backbone_time: 0.0266
neck_time: 0.0197
det_head_time: 0.0168
det_pse_time: 0.4697
FPS: 1.9
Testing 2/500
backbone_time: 0.0266
neck_time: 0.0197
det_head_time: 0.0171
det_pse_time: 0.4694
FPS: 1.9
Testing 3/500
backbone_time: 0.0266
neck_time: 0.0196
det_head_time: 0.0175
det_pse_time: 0.4691
FPS: 1.9
```
Evaluation on icdar2015
```bash
cd eval
sh eval_ic15.sh
```

Evaluation on total_text
```bash
cd eval
sh eval_tt.sh
```
### Prediction using pre training model

Put the file to be tested in the directory determined by the input parameter, run the following command, and save the output image in the directory determined by the output parameter

```bash
python3 predict.py ./config/psenet/psenet_r50_ic15_1024_Adam.py ./images ./out_img ./checkpoints/psenet_r50_ic15_1024_Adam/checkpoint_491_0.pdparams --report_speed
```

## 6 Code structure

### 6.1 structure

```
├─config                          
├─dataset                         
├─eval                           
├─models                          
├─results                         
├─utils                           
│  compile.sh                     
│  eval.py                        
│  init.sh                        
│  predict.py                     
│  README.md                      
│  README_cn.md                   
│  requirement.txt                
│  train.py                       
```

### 6.2 Parameter description

Parameters related to training and evaluation can be set in `train.py`, as follows:

|  Parameters   | default  | description | other |
|  ----  |  ----  |  ----  |  ----  |
| config| None, Mandatory| Configuration file path ||
| --checkpoint| None, Optional | Parameter path of pre training model ||
| --resume| None, Optional | Recovery training |For example: --resume checkpoint_44_0 is not an absolute path for breakpoint parameters. Note that|


### 6.3 Training process

#### Single machine training
```bash
python3 train.py $config_file
```

#### Multi machine training
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py $config_file
```

At this time, the program will import the output log of each process into the path of `./debug`:
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

#### Training output
After the training starts, you will get the output similar to the following. Each round of 'batch' training will print the current epoch, step and loss values.
```text
Epoch: [1 | 600]
(1/78) LR: 0.001000 | Batch: 6.458s | Total: 0min | ETA: 8min | Loss: 0.614 | Loss(text/kernel): 0.506/0.109 | IoU(text/kernel): 0.274/0.317 | Acc rec: 0.000
```

### 6.4 assessment process

```bash
python3 eval.py $config_file $pdparam_file --report_speed
```

The output is:
```
Testing 1/500
backbone_time: 0.0266
neck_time: 0.0197
det_head_time: 0.0168
det_pse_time: 0.4697
FPS: 1.9
Testing 2/500
backbone_time: 0.0266
neck_time: 0.0197
det_head_time: 0.0171
det_pse_time: 0.4694
FPS: 1.9
Testing 3/500
backbone_time: 0.0266
neck_time: 0.0196
det_head_time: 0.0175
det_pse_time: 0.4691
FPS: 1.9
```

### 6.5 Test process

```bash
python3 predict.py $config_file $input_dir $output_dir $pdparams_file --report_speed
```
At this time, the output result is saved in `$output_file`.


### 6.6 Prediction using pre training model

The process of prediction using the pre training model is as follows:

**step1:** Download pre training model
Google cloud：[https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing](https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing)
Corresponding relationship between model and configuration file:

|name|path|config|
| :---: | :---: | :---: |
|pretrain_1|psenet_r50_ic17_1024_Adam/checkpoint_33_0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_2|psenet_r50_ic17_1024_Adam/checkpoint_46_0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_3|psenet_r50_ic17_1260_Adam/checkpoint_68_0|[psenet_r50_ic17_1260.py](./config/psenet/psenet_r50_ic17_1260.py)|
|ic15_finetune_1|psenet_r50_ic15_1024_Adam/checkpoint_491_0|[finetune1.py](./config/psenet/finetune1.py)|
|ic15_finetune_2|psenet_r50_ic15_1260_Adam/best|[finetune2.py](./config/psenet/finetune2.py)|
|ic15_finetune_3|psenet_r50_ic15_1480_SGD/checkpoint_401_0|[finetune3.py](./config/psenet/finetune3.py)|
|tt_finetune_1|psenet_r50_tt/checkpoint_331_0|[psenet_r50_tt.py](./config/psenet/psenet_r50_tt.py)|
|tt_finetune_2|psenet_r50_tt/checkpoint_290_0|[psenet_r50_tt_finetune2.py](./config/psenet/psenet_r50_tt_finetune2.py)|

**step2:**Use the pre training model to complete the prediction
```bash
python3 predict.py $config_file $input_dir $output_dir $pdparams_file --report_speed
```
## 7 Model information

For other information about the model, please refer to the following table:

| information | description |
| --- | --- |
| Author | mingyuan xu、rongjie yi|
| Date | 2021.05 |
| Framework version | Paddle 2.0.2 |
| Application scenarios | Text detection |
| Support hardware | GPU、CPU |
| Download link | [Pre training model](https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing)  |
| Online operation | [botebook](https://aistudio.baidu.com/aistudio/projectdetail/1945560)、[Script](https://aistudio.baidu.com/aistudio/clusterprojectdetail/1796445)|
