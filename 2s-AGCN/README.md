# 2s-AGCN-Paddle

## 1 简介

This is the unofficial code based on **PaddlePaddle** of CVPR 2019 paper:

[Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf)

![模型结构图](https://github.com/ELKYang/2s-AGCN-paddle/blob/main/images/model_structure.png)

2s-AGCN是发表在CVPR2019上的一篇针对ST-GCN进行改进的文章，文章提出双流自适应卷积网络，针对原始ST-GCN的缺点进行了改进。在现有的基于GCN的方法中，图的拓扑是手动设置的，并且固定在所有图层和输入样本上。另外，骨骼数据的二阶信息（骨骼的长度和方向）对于动作识别自然是更有益和更具区分性的，在当时方法中很少进行研究。因此，文章主要提出一个基于骨架节点和骨骼两种信息融合的双流网络，并在图卷积中的邻接矩阵加入自适应矩阵，大幅提升骨骼动作识别的准确率，也为后续的工作奠定了基础（后续的骨骼动作识别基本都是基于多流的网络框架）。

论文地址：[2s-AGCN Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf)

原论文代码地址：[2s-AGCN Code](https://github.com/lshiwjx/2s-AGCN)

## 2 复现精度

> 在NTU-RGBD数据集上的测试效果如下

|                |    CS     |    CV     |
| :------------: | :-------: | :-------: |
| Js-AGCN(joint) |   85.8%   |  94.13%   |
| Bs-AGCN(bone)  |   86.7%   |   93.9%   |
|    2s-AGCN     | **88.5%** | **95.4%** |

在NTU-RGBD上达到验收标准：X-Sub=88.5%, X-View=95.1%

训练日志：[日志](https://github.com/ELKYang/2s-AGCN-paddle/tree/main/work_dir/ntu)

VisualDL可视化日志：[VDL](https://github.com/ELKYang/2s-AGCN-paddle/tree/main/runs)

模型权重：[model_weights](https://github.com/ELKYang/2s-AGCN-paddle/tree/main/weights)

## 3 数据集及数据预处理

1. 数据集地址：[NTU-RGBD](https://github.com/shahroudy/NTURGB-D)，下载后将其放到如下目录

   ```
   -data\  
     -nturgbd_raw\  
       -nturgb+d_skeletons\
    	  ...
       -samples_with_missing_skeletons.txt
   ```

2. 生成joint数据

   ```
   python data_gen/ntu_gendata.py
   ```

3. 生成bone数据

   ```
   python data_gen/gen_bone_data.py
   ```

## 4 环境依赖

- 硬件：Tesla V100 32G

- PaddlePaddle==2.2.2

- ```
  pip install -r requirements.txt
  ```

## 5 快速开始

1. **Clone本项目**

   ```
   # clone this repo
   git clone https://github.com/ELKYang/2s-AGCN-paddle.git
   cd 2s-AGCN-paddle
   ```

2. **模型训练**

   模型训练参数的配置文件均在`config`文件夹中(下面以**x-view**为例进行训练测试以及tipc)

   - `x-view joint`训练

     ```
     python main.py --config config/nturgbd-cross-view/train_joint.yaml
     ```

   - `x-view bone`训练

     ```
     python main.py --config config/nturgbd-cross-view/train_bone.yaml
     ```
     
     部分训练输出如下:
     
     ```
     [ Tue Apr 19 00:57:05 2022 ] Training epoch: 1
     100%|█████████████████████████████████████████| 588/588 [14:01<00:00,  1.43s/it]
     [ Tue Apr 19 01:11:07 2022 ] 	Mean training loss: 2.5229.
     [ Tue Apr 19 01:11:07 2022 ] 	Time consumption: [Data]01%, [Network]99%
     [ Tue Apr 19 01:11:07 2022 ] Eval epoch: 1
     100%|█████████████████████████████████████████| 296/296 [01:47<00:00,  2.97it/s]
     Accuracy:  0.5453729135854638  model:  ./runs/ntu_cv_agcn_bone
     [ Tue Apr 19 01:12:56 2022 ] 	Mean test loss of 296 batches: 1.4155468940734863.
     [ Tue Apr 19 01:12:56 2022 ] 	Top1: 54.54%
     [ Tue Apr 19 01:12:56 2022 ] 	Top5: 90.08%
     [ Tue Apr 19 01:12:56 2022 ] Training epoch: 2
     100%|█████████████████████████████████████████| 588/588 [14:03<00:00,  1.43s/it]
     [ Tue Apr 19 01:27:01 2022 ] 	Mean training loss: 1.3931.
     [ Tue Apr 19 01:27:01 2022 ] 	Time consumption: [Data]01%, [Network]99%
     [ Tue Apr 19 01:27:01 2022 ] Eval epoch: 2
     100%|█████████████████████████████████████████| 296/296 [01:47<00:00,  2.97it/s]
     Accuracy:  0.6540249313331925  model:  ./runs/ntu_cv_agcn_bone
     [ Tue Apr 19 01:28:50 2022 ] 	Mean test loss of 296 batches: 1.1467070579528809.
     [ Tue Apr 19 01:28:50 2022 ] 	Top1: 65.40%
     [ Tue Apr 19 01:28:50 2022 ] 	Top5: 92.53%
     ```

   训练完成后模型的VisualDL日志保存在`runs`文件夹

   模型参数，训练日志，训练配置等保存在`work_dir`文件夹

3. **模型测试**

   - `x-view joint`测试

     ```
     python main.py --config config/nturgbd-cross-view/test_joint.yaml --weights 'path to weghts'
     ```

   - `x-view bone`测试

     ```
     python main.py --config config/nturgbd-cross-view/test_bone.yaml --weights 'path to weights'
     ```
     
     测试输出如下
     
     ```
     [ Tue Apr 19 14:39:01 2022 ] Load weights from ./runs/ntu_cv_agcn_bone-49-29400.pdparams.
     [ Tue Apr 19 14:39:01 2022 ] Model:   paddle_model.agcn.Model.
     [ Tue Apr 19 14:39:01 2022 ] Weights: ./runs/ntu_cv_agcn_bone-49-29400.pdparams.
     [ Tue Apr 19 14:39:01 2022 ] Eval epoch: 1
     100%|███████████████████████████████████████████| 74/74 [01:52<00:00,  1.44s/it]
     Accuracy:  0.9387280794422143  model:  ./runs/ntu_cv_agcn_test_bone
     [ Tue Apr 19 14:40:55 2022 ] 	Mean test loss of 74 batches: 0.23004150390625.
     [ Tue Apr 19 14:40:55 2022 ] 	Top1: 93.87%
     [ Tue Apr 19 14:40:55 2022 ] 	Top5: 99.09%
     [ Tue Apr 19 14:40:55 2022 ] Done.
     ```

4. **模型预测**

   这里使用x-view测试集中的10条数据用来做预测

   ```
   python main.py --config config/nturgbd-cross-view/test_joint_lite.yaml --weights 'path to weights'
   ```

   预测结果如下(详细的预测信息生成在`work_dir/ntu/xview/agcn_test_joint_lite`文件夹下)：

   ```
   predict action index:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
   true action index:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   100%|█████████████████████████████████████████████| 1/1 [00:02<00:00,  2.21s/it]
   Accuracy:  1.0  model:  ./runs/ntu_cv_agcn_test_joint_lite
   [ Mon Apr 18 23:40:28 2022 ] 	Mean test loss of 1 batches: 0.004666340071707964.
   [ Mon Apr 18 23:40:28 2022 ] 	Top1: 100.00%
   [ Mon Apr 18 23:40:28 2022 ] 	Top5: 100.00%
   [ Mon Apr 18 23:40:28 2022 ] Done.
   ```

5. **模型动转静推理**

   - 模型动转静
   
     ```
     python export_model.py --save_dir ./output --model_path 'The path of model for export' --batch 10
     ```
   
     > batch是在静态推理时使用的批大小，需要与infer阶段一致；
     >
     > 会在output文件夹生成静态模型：
     >
     >    |-- output
     >       |-- model.pdipaprams
     >       |-- model.pdipaprams.info
     >       |-- model.pdmodel
   
   - 生成小数据集
   
     模型推理时使用小数据集进行模型推理，以使用xsub的joint数据生成tiny dataset：
   
     ```
     pyhton ./data_gen/gen_infer_sample_data.py --dataset 'xsub' --mode 'joint' --data_num 50
     ```
   
   - 模型静态推理
   
     > 安装auto_log，需要进行安装，安装方式如下：
     >
     > ```
     > git clone https://github.com/LDOUBLEV/AutoLog
     > cd AutoLog/
     > pip3 install -r requirements.txt
     > python3 setup.py bdist_wheel
     > pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
     > ```
   
     进行模型的静态推理
   
     ```
     pyhton infer.py --data_file 'path to tiny data set' --label_file 'path to tiny label set' --model_file ./output/model.pdmodel --params_file ./output/model.pdiparams
     ```
   
     静态推理部分主要输出：
   
     ```
     Batch action class Predict:  [0 1 2 3 4 5 6 7 8 9] Batch action class True:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] Batch Accuracy:  1.0 Batch sample Name:  ['S001C001P001R001A001.skeleton', 'S001C001P001R001A002.skeleton', 'S001C001P001R001A003.skeleton', 'S001C001P001R001A004.skeleton', 'S001C001P001R001A005.skeleton', 'S001C001P001R001A006.skeleton', 'S001C001P001R001A007.skeleton', 'S001C001P001R001A008.skeleton', 'S001C001P001R001A009.skeleton', 'S001C001P001R001A010.skeleton']
     Batch action class Predict:  [10 11 12 13 14 15 15 17 18 19] Batch action class True:  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] Batch Accuracy:  0.9 Batch sample Name:  ['S001C001P001R001A011.skeleton', 'S001C001P001R001A012.skeleton', 'S001C001P001R001A013.skeleton', 'S001C001P001R001A014.skeleton', 'S001C001P001R001A015.skeleton', 'S001C001P001R001A016.skeleton', 'S001C001P001R001A017.skeleton', 'S001C001P001R001A018.skeleton', 'S001C001P001R001A019.skeleton', 'S001C001P001R001A020.skeleton']
     Batch action class Predict:  [20 21 22 23 24 25 26 27 28 29] Batch action class True:  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29] Batch Accuracy:  1.0 Batch sample Name:  ['S001C001P001R001A021.skeleton', 'S001C001P001R001A022.skeleton', 'S001C001P001R001A023.skeleton', 'S001C001P001R001A024.skeleton', 'S001C001P001R001A025.skeleton', 'S001C001P001R001A026.skeleton', 'S001C001P001R001A027.skeleton', 'S001C001P001R001A028.skeleton', 'S001C001P001R001A029.skeleton', 'S001C001P001R001A030.skeleton']
     Batch action class Predict:  [30 31 32 33 34 35 36 37 44 39] Batch action class True:  [30, 31, 32, 33, 34, 35, 36, 37, 38, 39] Batch Accuracy:  0.9 Batch sample Name:  ['S001C001P001R001A031.skeleton', 'S001C001P001R001A032.skeleton', 'S001C001P001R001A033.skeleton', 'S001C001P001R001A034.skeleton', 'S001C001P001R001A035.skeleton', 'S001C001P001R001A036.skeleton', 'S001C001P001R001A037.skeleton', 'S001C001P001R001A038.skeleton', 'S001C001P001R001A039.skeleton', 'S001C001P001R001A040.skeleton']
     Batch action class Predict:  [40 41 42 43 44 45 46 47 48 49] Batch action class True:  [40, 41, 42, 43, 44, 45, 46, 47, 48, 49] Batch Accuracy:  1.0 Batch sample Name:  ['S001C001P001R001A041.skeleton', 'S001C001P001R001A042.skeleton', 'S001C001P001R001A043.skeleton', 'S001C001P001R001A044.skeleton', 'S001C001P001R001A045.skeleton', 'S001C001P001R001A046.skeleton', 'S001C001P001R001A047.skeleton', 'S001C001P001R001A048.skeleton', 'S001C001P001R001A049.skeleton', 'S001C001P001R001A050.skeleton']
     Infer Mean Accuracy:  0.96
     ```
   
6. **双流融合生成结果**

   ```
   python ensemble.py --datasets ntu/view
   ```
   
   双流融合输出如下所示：
   
   ```
   100%|██████████████████████████████████| 18932/18932 [00:00<00:00, 54076.63it/s]
   acc:  0.954  acc5:  0.993
   ```

## 6 TIPC
运行下述命令，完成训推一体化脚本测试
```
# 准备tipc数据
bash test_tipc/prepare.sh ./test_tipc/configs/2s-AGCN/train_infer_python.txt 'lite_train_lite_infer'
# 开启训推一体'lite_train_lite_infer'模式
bash test_tipc/test_train_inference_python.sh test_tipc/configs/2s-AGCN/train_infer_python.txt 'lite_train_lite_infer'
```
详细输出见`test_tipc`下的[README.md](https://github.com/ELKYang/2s-AGCN-paddle/blob/main/test_tipc/README.md)文档

## 7 代码结构与详细说明
```
|-- paddle_2s_AGCN
   |-- config                       # 模型训练所需的yaml配置
   |-- data_gen                     # 数据预处理文件
      |-- __init__.py
      |-- gen_bone_data.py          # 获取训练所需骨骼数据
      |-- gen_infer_sample_data.py  # 生成推理数据
      |-- gen_motion_data.py
      |-- ntu_gendata.py            # NTU-RGB-D完整数据集预处理
      |-- preprocess.py
      |-- rotation.py
   |-- feeders                      # 读取数据集内数据
      |-- __init__.py
      |-- feeder.py                 # 创建paddle.io.Dataset
      |-- tools.py
   |-- graph                        # 生成骨骼拓扑图
      |-- ntu_rgb_d.py              # 生成NTU-RGB-D骨骼拓扑图
      |-- tools.py
   |-- output                       # 存放静态模型以及AutoLog日志文件
      |-- model.pdipaprams
      |-- model.pdipaprams.info
      |-- model.pdmodel
   |-- paddle_model                 # paddle模型定义
      |-- __init__.py
      |-- agcn.py                   # AGCN模型
   |-- runs                         # VisualDL日志文件夹
   |-- work_dir                     # 模型训练日志文件夹
   |-- weights                      # 权重文件夹
   |-- test_tipc                    # TIPC训推一体化认证
   |-- README.md
   |-- ensemble.py                  # 双流集成代码
   |-- export_model.py              # 导出静态模型
   |-- main.py                      # 单卡训练测试代码
   |-- requirements.txt             # 环境配置文件
```
## 8 附录
**相关信息**
| 信息     | 描述                                                         |
| -------- | ------------------------------------------------------------ |
| 作者     | [kunkun0w0](https://github.com/kunkun0w0)、[ELKYang](https://github.com/ELKYang) |
| 日期     | 2022年4月                                                    |
| 框架版本 | PaddlePaddle==2.2.0                                          |
| 应用场景 | 骨架动作识别                                                 |
| 硬件支持 | GPU, CPU                                                     |
| AIStudio地址 | [Notebook](https://aistudio.baidu.com/aistudio/projectdetail/3821064?contributionType=1&shared=1) |

代码参考：https://github.com/lshiwjx/2s-AGCN

感谢百度飞浆团队提供的算力支持！
