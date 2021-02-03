简体中文 | [English](README_en.md)

# ECO视频分类模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [参考论文](#参考论文)
- [其它参考](#其它参考)


## 模型简介

ECO是视频分类领域的高精度模型，使用2D和3D两个分支。2D分支捕捉视频中空间维度的信息。3D分支捕获视频中时间维度的信息，最终将两个分支的特征拼接得到预测结果。

<p align="center">
<img src="images/eco.png" height=300 width=500 hspace='10'/> <br />
ECO Overview
</p>

详细内容请参考ECCV 2018论文[ECO: Efficient Convolutional Network for Online Video Understanding](https://arxiv.org/abs/1804.09066)


## 数据准备

ECO模型的训练数据采用UCF-101数据集。

*  方法1：
原始数据请在[UCF-101数据集](https://www.crcv.ucf.edu/data/UCF101.php)下载

下载到data文件夹后可以运行如下命令进行数据预处理：

```bash
python avi2jpg.py

python jpg2pkl.py

python data_list_gener.py
```
*  方法2(建议采用该方法)：

直接下载视频转为jpg图片的[图片数据集](https://aistudio.baidu.com/aistudio/datasetdetail/52155)并解压，然后运行如下命令：

```bash
python jpg2pkl.py

python data_list_gener.py
```

## 模型训练

数据准备完成后，可通过如下方式启动训练：

```bash
python train.py --epoch 5 --use_gpu True --pretrain True

```
- 通过 `-pretrain`参数，您可以下载我们训练好的模型进行训练微调[Baidu Pan](https://pan.baidu.com/s/1yU3TILs-39CCPWuBD8NqHg) code: h47v


### 训练资源要求

*  单卡V100，seg\_num=12, batch\_size=12，显存占用约30G。


## 模型测试

可通过如下命令进行模型测试:

```bash
python test.py --weights 'trained_model/eco_1_8.pdparams'
```

- 通过 `--weights`参数指定待测试模型文件的路径，您可以下载我们训练好的模型进行测试[Baidu Pan](https://pan.baidu.com/s/1yU3TILs-39CCPWuBD8NqHg) code: h47v

在UCF-101数据集下评估精度如下:

| Acc1 | 
| :---: | 
| 97.57 | 


## 参考论文

- [ECO: Efficient Convolutional Network for Online Video Understanding](https://arxiv.org/abs/1804.09066), Mohammadreza Zolfaghari, Kamaljeet Singh, Thomas Brox 

## 其它参考

- [AI Studio 项目链接](https://aistudio.baidu.com/aistudio/projectdetail/674555)