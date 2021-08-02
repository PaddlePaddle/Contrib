# MAML-paddle
## 一、简介

Model-Agnostic Meta-Learning（MAML）算法是一种模型无关的元学习算法，其模型无关体现在，能够与任何使用了梯度下降法的模型相兼容，广泛应用于各种不同的机器学习任务，包括分类、识别、强化学习等领域。

元学习的目标，是在大量不同的任务上训练一个模型，使其能够使用极少量的训练数据（即小样本），进行极少量的梯度下降步数，就能够迅速适应新任务，解决新问题。

在本项目复现的文献中，通过对模型参数进行显式训练，从而获得在各种任务下均能良好泛化的模型初始化参数。当面临小样本的新任务时，使用该初始化参数，能够在单步（或多步）梯度更新后，实现对该任务的学习和适配。为了复现文献中的实验结果，本项目基于paddlepaddle深度学习框架，在omniglot数据集上进行训练和测试，目标是达到并超过原文献的模型性能。

论文 ：

Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

https://arxiv.org/abs/1703.03400

## 二、复现精度

基于paddlepaddle深度学习框架，对文献MAML进行复现后，汇总各小样本任务下的测试精度，如下表所示。

|任务|Test ACC|
|----|----|
|5-way-1-shot|99.2%|
|5-way-5-shot|99.5%|
|20-way-1-shot|95.0%|
|20-way-5-shot|98.7%|

超参数配置如下表所示：

|超参数名|设置值|
|----|----|
|batch_size|32|
|update_step|5|
|update_step_test|5|
|meta_lr|0.001|
|base_lr|0.1|

## 三、数据集

本项目使用的是Omniglot数据集。

Omniglot 数据集包含50个不同的字母表，每个字母表中的字母各包含20个手写字符样本，每一个手写样本都是不同的人通过亚马逊的 Mechanical Turk 在线绘制的。Omniglot数据集的多样性强于MNIST数据集，是增强版的MNIST，常用与小样本识别任务。

![](https://ai-studio-static-online.cdn.bcebos.com/cd4e94bac7c3470e800cfb3426fdf8954c5c108ac9504b658be492b034a2fb6b)

数据集链接：
https://aistudio.baidu.com/aistudio/datasetdetail/78550

## 四、环境依赖
- 硬件：
    - x86 cpu
    - NVIDIA GPU
- 框架：
    - PaddlePaddle >= 2.0.0

- 其他依赖项：
    - numpy==1.19.3
    - opencv_python==4.5.1.48
    - tqdm==4.59.0

## 五、快速开始
1、首先解压数据集，并将images_background和images_evaluation路径下的内容，拷贝到“data/omniglot/”中。

2、执行make_data.py，对图像数据进行遍历、处理，构建训练集、验证集和测试集的numpy格式数据，并保存到工程根目录下。

3、可以通过以下代码，打开并显示四个样本；
```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
# 加载训练集和测试集
x_train = np.load('omniglot_train.npy')  # (964, 20, 1, 28, 28)
plt.subplot(1,4,1)
plt.imshow(x_train[0,0,0,:,:], cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1,4,2)
plt.imshow(x_train[1,0,0,:,:], cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1,4,3)
plt.imshow(x_train[2,0,0,:,:], cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1,4,4)
plt.imshow(x_train[3,0,0,:,:], cmap=plt.cm.gray)
plt.axis('off')
```
4、执行以下命令启动训练：

`python train.py --n_way 5 --k_spt 1 --use_gpu`执行5 way 1 shot训练

`python train.py --n_way 5 --k_spt 5 --use_gpu`执行5 way 5 shot训练

`python train.py --n_way 20 --k_spt 1 --use_gpu`执行20 way 1 shot训练

`python train.py --n_way 20 --k_spt 5 --use_gpu`执行20 way 5 shot训练

训练文件参数如下：

|参数选项|默认值|说明|
|----|----|----|
|--n_way|5|小样本任务类别数|
|--k_spt|1|小样本任务每个支持集类别的样本数|
|--k_query|15|小样本任务每个类别测试的无标签样本数|
|--task_num|32|训练时，一个batch的任务数|
|--glob_update_step|5|全局更新步长|
|--glob_update_step_test|5|全局更新步长（测试）|
|--glob_meta_lr|0.001|全局元学习率|
|--glob_base_lr|0.1|全局基学习率|
|--epochs|10000|训练epoch的轮数|
|--use_gpu|true|是否使用gpu|

5、执行以下命令启动评估：

`python evaluate.py --n_way 5 --k_spt 1 --use_gpu`执行5 way 1 shot评估

`python evaluate.py --n_way 5 --k_spt 5 --use_gpu`执行5 way 5 shot评估

`python evaluate.py --n_way 20 --k_spt 1 --use_gpu`执行20 way 1 shot评估

`python evaluate.py --n_way 20 --k_spt 5 --use_gpu`执行20 way 5 shot评估

## 六、代码结构与详细说明
### 6.1 算法框架
考虑一个关于任务T的分布p(T)，我们希望模型能够对该任务分布很好的适配。在K-shot（即K个学习样本）的学习任务下，从p(T)分布中随机采样一个新任务Ti，在任务Ti的样本分布qi中随机采样K个样本，用这K个样本训练模型，获得LOSS，实现对模型f的内循环更新。然后再采样query个样本，评估新模型的LOSS，然后对模型f进行外循环更新。反复上述过程，从而使最终模型能够对任务分布p(T)上的所有情况，能够良好地泛化。算法可用下图进行示意。

 ![](https://ai-studio-static-online.cdn.bcebos.com/5c1cc7e52f7e4a3d98b9693a6e27309c72d041b310994d889640a29221e47c52)
 
### 6.2 算法流程
MAML算法针对小样本图像分类任务的计算流程，如下图所示：
 
 ![](https://ai-studio-static-online.cdn.bcebos.com/bd44f95ed7564189a010b04367f79ba15362cbf1dc9c491ea539ffb2b06dfa23)
 
本项目的难点在于，算法包含外循环和内循环两种梯度更新方式。内循环针对每一种任务T进行梯度更新，用更新后的模型重新评估LOSS；而外循环则要使用内循环中更新后的LOSS，在所有任务上更新原始模型。
使用paddle经典的动态图框架，在内循环更新完成后，模型各节点参数已经发生变化，loss已无法反传到先前的模型参数上。外循环的参数更新公式为

![](https://ai-studio-static-online.cdn.bcebos.com/2bf80b14ecee42d88b19eceae07ce5fa4a7d15f1cf764ce89ddee5b719270f51)


这里，要使用θ_i^'参数模型计算的LOSS，反传回θ，使用经典动态图模型架构无法实现。本方案通过自定义参数的方式，使函数层层级联，实现更灵活的参数控制。

## 七、模型信息
训练完成后，最优模型保存在model目录下。

| 信息 | 说明 |
| --- | --- |
| 发布者 | hrdwsong |
| 时间 | 2021.06 |
| 框架版本 | Paddle 2.0.1 |
| 应用场景 | 小样本学习、元学习 |
| 支持硬件 | GPU、CPU |
| 在线运行 | https://aistudio.baidu.com/aistudio/projectdetail/1869590 |