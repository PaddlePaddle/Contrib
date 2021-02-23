# representation-flow with paddlepaddle

---
## 内容

- [模型简介](#模型简介)
- [准备工作](#准备工作)
- [模型训练](#模型训练)
- [复现精度](#复现精度)
- [模型预测](#模型预测)
- [参考论文](#参考论文)

#### 模型介绍
[CVPR19]Representation Flow for Action Recognition：双流网络中光流计算量较大，且光流生产需要保存到硬盘，这样不能做到实时的动作识别。Representation Flow是一个可微分的网络层迭代计算光流信息，并且可以通过Flow of Flow堆叠多个光流层来提升性能。

网络结构
![网络结构](https://github.com/Qdriving/Contrib/blob/master/representation-flow/flow.png)

算法描述

![算法描述](https://github.com/Qdriving/Contrib/blob/master/representation-flow/alg.png)

#### 准备工作
1，安装FFmpeg和Lintel
   请单独下载并安装FFmpeg和Lintel，配置好环境变量，具体请参考百度内容；然后进行下一步。

2，下载并解压hmdb数据集到你的目录，参考地址：https://aistudio.baidu.com/aistudio/datasetdetail/48783。

3，参照一下命令，将avi格式文件转换为mp4文件方便lintel读取：
```bash
$python avitomp4.py -data_dir "你自己的数据目录"
```

#### 模型训练
```bash
$python train_model.py -mode='rgb' -exp_name='train2dfof' -learnable='[1,1,1,1]' -niter=2 -model='2d' -system='hmdb' -batch_size 12 -learning_rate 1e-2 -momentum 0.9 
```
使用optimizer.MomentumOptimizer优化器lr=1e-2, momentum=0.9 训练20个epoch后训练集精度达到0.99以上，再用lr=2e-3, momentum=0.4训练7个epoch，测试集精度在0.89左右。
环境参考：
   CPU | 4
   RAM | 32GB
   GPU | v100
   显存 | 16GB
   磁盘 | 100GB
  

#### 模型预测
通过以下方式预测模型
```bash
$python test_model.py -mode='rgb' -exp_name='eval2dfof' -learnable='[1,1,1,1]' -niter=2 -model='2d' -system='hmdb' -batch_size 128  -check_point pretrained
```

#### 复现精度
Model|Acc1
---|---
Representation-Flow|0.89

#### 参考论文
https://arxiv.org/pdf/1810.01455 


