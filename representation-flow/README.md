# representation-flow

#### 介绍
[CVPR19]Representation Flow for Action Recognition：双流网络中光流计算量较大，且光流生产需要保存到硬盘，这样不能做到实时的动作识别。Representation Flow是一个可微分的网络层迭代计算光流信息，并且可以通过Flow of Flow堆叠多个光流层来提升性能。

网络结构
![网络结构](https://github.com/Qdriving/Contrib/blob/master/representation-flow/flow.png)

算法描述

![算法描述](https://github.com/Qdriving/Contrib/blob/master/representation-flow/alg.png)

#### 环境、数据准备
1，安装FFmpeg和Lintel
   请单独下载并安装FFmpeg和Lintel，然后进行下一步。
2，将avi格式文件转换为mp4文件方便lintel读取：
```bash
$python avitomp4.py
```

#### 模型训练&预测
```bash
$python train_model.py -mode='rgb' -exp_name='train2dfof' -learnable='[1,1,1,1]' -niter=2 -model='2d' -system='hmdb' -batch_size 12 -learning_rate 1e-2 -momentum 0.9 
```
使用optimizer.MomentumOptimizer优化器lr=1e-2, momentum=0.9 训练20个epoch后训练集精度达到0.99以上，再用lr=2e-3, momentum=0.4训练7个epoch，测试集精度在0.89左右。

通过以下方式预测模型
```bash
$python test_model.py -mode='rgb' -exp_name='eval2dfof' -learnable='[1,1,1,1]' -niter=2 -model='2d' -system='hmdb' -batch_size 128  -check_point pretrained
```

#### 精度对比

![精度对比](https://github.com/Qdriving/Contrib/blob/master/representation-flow/acc.png)

