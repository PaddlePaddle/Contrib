# PSENet

[English](./README.md) | 简体中文
   
   * [PSENet](#resnet)
      * [一、简介](#一简介)
      * [二、复现精度](#二复现精度)
      * [三、数据集](#三数据集)
      * [四、环境依赖](#四环境依赖)
      * [五、快速开始](#五快速开始)
         * [step1: 训练](#step1-训练)
         * [step2: 评估](#step2-评估)
         * [step3: 测试](#step3-测试)
      * [六、代码结构与详细说明](#六代码结构与详细说明)
         * [6.1 代码结构](#61-代码结构)
         * [6.2 参数说明](#62-参数说明)
         * [6.3 训练流程](#63-训练流程)
            * [单机训练](#单机训练)
            * [多机训练](#多机训练)
            * [训练输出](#训练输出)
         * [6.4 评估流程](#64-评估流程)
         * [6.5 测试流程](#65-测试流程)
         * [6.6 使用预训练模型预测](#66-使用预训练模型预测)
      * [七、模型信息](#七模型信息)

## 一、简介

本项目基于paddlepaddle框架复现PSENet，PSENet是一种新的实例分割网络，它有两方面的优势。 首先，psenet作为一种基于分割的方法，能够对任意形状的文本进行定位。其次，该模型提出了一种渐进的尺度扩展算法，该算法可以成功地识别相邻文本实例。


**论文:**
- [1] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao. Shape robust text detection with progressive scale expansion network. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 9336–9345, 2019.<br>

**参考项目：**
- [https://github.com/whai362/PSENet](https://github.com/whai362/PSENet)

**项目aistudio地址：**
- notebook任务：[https://aistudio.baidu.com/aistudio/projectdetail/1945560](https://aistudio.baidu.com/aistudio/projectdetail/1945560)
- 脚本任务：[https://aistudio.baidu.com/aistudio/clusterprojectdetail/1796445](https://aistudio.baidu.com/aistudio/clusterprojectdetail/1796445)

## 二、复现精度

>该列指标在ICDAR2015的测试集测试

train from scratch细节：


| |epoch|opt|short_size|batch_size|dataset|memory|card|precision|recall|hmean|FPS|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|pretrain_1|33|Adam|1024|16|ICDAR2017|32G|1|0.68290|0.68850|0.68569|5.0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_2|46|Adam|1024|16|ICDAR2013、ICDAR2017、COCO_TEXT|32G|4|0.69678|0.69812|0.69745|5.0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_3|68|Adam|1260|16|ICDAR2013、ICDAR2015、ICDAR2017、COCO_TEXT|32G|1|0.86526|0.80693|0.83508|2.0|[psenet_r50_ic17_1260.py](./config/psenet/psenet_r50_ic17_1260.py)|


**ICDAR2015**
>该列指标在ICDAR2015的测试集测试

训练细节：

| |pretrain|epoch|opt|short_size|batch_size|dataset|memory|card|precision|recall|hmean|FPS|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|finetune_1|pretrain_1|491|Adam|1024|16|ICDAR2015|32G|1|0.86463|0.80260|0.83246|5.0|[finetune1.py](./config/psenet/finetune1.py)|
|finetune_2|pretrain_3|-|Adam|1260|16|ICDAR2015|32G|1|0.87024|0.81367|0.84101|2.0|[finetune2.py](./config/psenet/finetune2.py)|
|finetune_3|finetune_2|401|SGD|1480|16|ICDAR2015|32G|1|<font color='red'>0.88060</font>|<font color='red'>0.82378</font>|<font color='red'>0.85124</font>|<font color='red'>1.8</font>|[finetune3.py](./config/psenet/finetune3.py)|

**Total_text**
>该列指标在Total_Text的测试集测试

训练细节：

| |pretrain|epoch|opt|short_size|batch_size|dataset|memory|card|precision|recall|hmean|FPS|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|finetune_1|None|331|Adam|736|16|Total_Text|32G|1|0.84823|0.76007|0.80173|10.1|[psenet_r50_tt.py](./config/psenet/psenet_r50_tt.py)|
|finetune_2|pretrain_2|290|Adam|736|16|Total_Text|32G|1|<font color='red'>0.88482</font>|<font color='red'>0.79002</font>|<font color='red'>0.83474</font>|<font color='red'>10.1</font>|[psenet_r50_tt_finetune2.py](./config/psenet/psenet_r50_tt_finetune2.py)|

**模型下载**
模型地址：[谷歌云盘](https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing)

模型对应：

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

## 三、数据集

[ICDAR2015文本检测数据集](https://rrc.cvc.uab.es/?ch=4&com=downloads)。

- 数据集大小：
  - 训练集：1000张
  - 测试集：500张
- 数据格式：矩形框文本数据集

[Total Text文本检测数据集](https://github.com/cs-chan/Total-Text-Dataset)。

- 数据集大小：
  - 训练集：1255张
  - 测试集：300张
- 数据格式：弯曲文本数据集

## 四、环境依赖

- 硬件：GPU、CPU

- 框架：
  - PaddlePaddle >= 2.0.0

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/PaddlePaddle/Contrib.git
cd PSENet
export PYTHONPATH=./
```
**安装依赖**
```bash
sh init.sh
```

**编译pse**
因为原作者提供了cpp的pse代码加速后处理的过程，所以需要编译，注意在aistudio上我们已经打包了编译好的，不需要执行这一步~
```bash
sh compile.sh
```
如果实在编译不成功可以使用提供的python版本的pse代码：
[pypse.py](./models/pypse.py)


### step2: 训练
```bash
python3 train.py ./config/psenet/psenet_r50_ic15_1024_Adam.py
```
如果训练中断通过 --resume 参数恢复，例如使用上述命令在第44epoch第0iter中断则：
```bash
python3 train.py ./config/psenet/psenet_r50_ic15_1024_Adam.py --resume checkpoint_44_0 # 不是断点参数的绝对路径请注意
```
如果你想分布式训练并使用多卡：
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py config/psenet/psenet_r50_ic17_1024_Adam.py
```

此时的输出为：
```
Epoch: [1 | 600]
(1/78) LR: 0.001000 | Batch: 6.458s | Total: 0min | ETA: 8min | Loss: 0.614 | Loss(text/kernel): 0.506/0.109 | IoU(text/kernel): 0.274/0.317 | Acc rec: 0.000
```
由于是目标检测任务，需要关注 ``loss`` 逐渐降低，``IOU`` 逐渐升高。

### step3: 测试
```bash
python3 eval.py ./config/psenet/psenet_r50_ic15_1024_Adam.py ./checkpoints/psenet_r50_ic15_1024_Adam/checkpoint_491_0.pdparams --report_speed
```
此时的输出为：
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
ICDAR2015评估
```bash
cd eval
sh eval_ic15.sh
```

total_text评估(注意tt数据集的评估需要使用python2，如果你在aistudio上使用我们的项目我们提供了打包的python可以直接执行详情见eval_tt.sh文件)
```bash
cd eval
sh eval_tt.sh
```
### 使用预训练模型预测

将需要测试的文件放在参数input确定的目录下， 运行下面指令，输出图片保存在output参数确定的目录下

```bash
python3 predict.py ./config/psenet/psenet_r50_ic15_1024_Adam.py ./images ./out_img ./checkpoints/psenet_r50_ic15_1024_Adam/checkpoint_491_0.pdparams --report_speed
```

## 六、代码结构与详细说明

### 6.1 代码结构

```
├─config                          # 配置
├─dataset                         # 数据集加载
├─eval                            # 评估脚本
├─models                          # 模型
├─results                         # 可视化结果
├─utils                           # 工具代码
│  compile.sh                     # 编译pse.cpp
│  eval.py                        # 评估
│  init.sh                        # 安装依赖
│  predict.py                     # 预测
│  README.md                      # 英文readme
│  README_cn.md                   # 中文readme
│  requirement.txt                # 依赖
│  train.py                       # 训练
```

### 6.2 参数说明

可以在 `train.py` 中设置训练与评估相关参数，具体如下：

|  参数   | 默认值  | 说明 | 其他 |
|  ----  |  ----  |  ----  |  ----  |
| config| None, 必选| 配置文件路径 ||
| --checkpoint| None, 可选 | 预训练模型参数路径 ||
| --resume| None, 可选 | 恢复训练 |例如：--resume checkpoint_44_0 不是断点参数的绝对路径请注意|


### 6.3 训练流程

#### 单机训练
```bash
python3 train.py $config_file
```

#### 多机训练
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py $config_file
```

此时，程序会将每个进程的输出log导入到`./debug`路径下：
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

#### 训练输出
执行训练开始后，将得到类似如下的输出。每一轮`batch`训练将会打印当前epoch、step以及loss值。
```text
Epoch: [1 | 600]
(1/78) LR: 0.001000 | Batch: 6.458s | Total: 0min | ETA: 8min | Loss: 0.614 | Loss(text/kernel): 0.506/0.109 | IoU(text/kernel): 0.274/0.317 | Acc rec: 0.000
```

### 6.4 评估流程

```bash
python3 eval.py $config_file $pdparam_file --report_speed
```

此时的输出为：
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

### 6.5 测试流程

```bash
python3 predict.py $config_file $input_dir $output_dir $pdparams_file --report_speed
```
此时的输出结果保存在`$output_dir`下面


### 6.6 使用预训练模型预测

使用预训练模型预测的流程如下：

**step1:** 下载预训练模型
谷歌云盘：[https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing](https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing)
模型与配置文件对应关系：

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

**step2:** 使用预训练模型完成预测
```bash
python3 predict.py $config_file $input_dir $output_dir $pdparams_file --report_speed
```
## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | 徐铭远、衣容颉|
| 时间 | 2021.05 |
| 框架版本 | Paddle 2.0.2 |
| 应用场景 | 文本检测 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [预训练模型](https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing)  |
| 在线运行 | [botebook](https://aistudio.baidu.com/aistudio/projectdetail/1945560)、[脚本任务](https://aistudio.baidu.com/aistudio/clusterprojectdetail/1796445)|
