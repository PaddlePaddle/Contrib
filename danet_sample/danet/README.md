# 基于paddlepaddle的DANet复现代码

## 文件结构说明
*  dataset：数据预处理代码dataset_preprocess.py，生成数据集文件列表；数据集读取构建器myDataset.py，用于读取数据
*  my_utils:配置文件global_config.py，设置各种路径及参数；常用工具com_utils.py，主要用于显示各种进度
*  networks:各种网络结构
*  net_train.py：训练脚本
*  net_train.py：测试脚本
*  net_train.py：推理脚本
*  requirement.txt：环境配准需求

## 运行说明
以Cityscape-gt数据集为例：
1. 运行cityscapesscripts/preparation/createTrainIdLabelImgs.py生成trainid文件
2. 运行danet/my_utils/global_config.py，进行文件结构检查
3. 运行danet/dataset/dataset_preprocess.py，生成数据集列表
4. 运行danet/net_train.py，进行训练，得到训练曲线及模型参数
5. 运行danet/net_test.py，进行测试，得到测试集混淆矩阵及iou指标
6. 运行danet/net_infer.py，进行推理，得到推理结果
