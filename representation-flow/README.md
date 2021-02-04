# representation-flow

#### 介绍
[CVPR19]Representation Flow for Action Recognition：双流网络中光流计算量较大，且光流生产需要保存到硬盘，这样不能做到实时的动作识别。Representation Flow是一个可微分的网络层迭代计算光流信息，并且可以通过Flow of Flow堆叠多个光流层来提升性能。

网络结构
![输入图片说明](https://images.gitee.com/uploads/images/2021/0203/144029_c54f5a91_5371233.png "屏幕截图.png")

算法描述

![输入图片说明](https://images.gitee.com/uploads/images/2021/0203/144454_92cda52b_5371233.png "屏幕截图.png")

#### 环境、数据准备
1，安装FFmpeg和Lintel
  在.bashrc文件里面添加以下内容并source
```bash
export ffmpegpath=/home/aistudio/work/FFmpeg
export PATH=${ffmpegpath}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${ffmpegpath}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CPATH=${ffmpegpath}/include${CPATH:+:${CPATH}}
export LIBRARY_PATH=${ffmpegpath}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
```
运行以下程序安装配置环境：
```bash
$python envprepare.py
```
将avi格式文件转换为mp4文件方便lintel读取：
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

