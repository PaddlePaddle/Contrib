# MARS 视频分类模型动态图实现

---
## 内容

- [模型简介](#模型简介)
- [准备工作](#准备工作)
- [模型训练](#模型训练)
- [复现精度](#复现精度)
- [模型预测](#模型预测)
- [参考论文](#参考论文)

## 模型简介
在MARS(MARS: Motion-Augmented RGB Stream for ActionRecognition)这篇论文中作者提出了两种新的学习策略，即基于蒸馏的概念和在特权信息下的学习，以避免在测试时进行光流计算，同时保留双流方法的性能。

详细内容请参考CVPR 2019论文[MARS: Motion-Augmented RGB Stream for ActionRecognition](https://hal.inria.fr/hal-02140558)

## 准备工作
 数据集：数据使用HMDB51，可以在https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/ 下载HMDB51数据集

**安装带GPU支持的opencv**
```bash
wget https://github.com/opencv/opencv/archive/4.3.0.tar.gz -O opencv-4.3.0.tar.gz
wget https://github.com/opencv/opencv_contrib/archive/4.3.0.tar.gz -O opencv_contrib-4.3.0.tar.gz
tar -xzvf opencv-4.3.0.tar.gz
tar -xzvf opencv_contrib-4.3.0.tar.gz
cd opencv-4.3.0
mkdir build
mkdir install
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=../install \ #指定安装路径
      -D CUDA_ARCH_BIN='7.5' \ #指定GPU算力，在NVIDIA官网查询
      -D WITH_CUDA=ON \ #使用CUDA
      -D WITH_CUBLAS=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.3.0/modules \ #opencv_contrib modules路径
``` 
**从视频中提取帧可以使用**
```bash
python utils1/extract_frames.py path_to_video_files path_to_extracted_frames start_class end_class
```  
**从视频中提取帧+光流可以使用**
```bash
export OPENCV=path_where_opencv_is_installed

g++ -std=c++11 tvl1_videoframes.cpp -o tvl1_videoframes -I${OPENCV}include/opencv4/ -L${OPENCV}lib64 -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudaarithm

python utils1/extract_frames_flows.py path_to_video_files path_to_extracted_flows_frames start_class end_cl
```
**下载预训练模型**
需要从https://drive.google.com/drive/folders/1OVhBnZ_FmqMSj6gw9yyrxJJR8yRINb_G?usp=sharing 下载预训练的pytorch模型

**预训练模型转化**
将pytorch预训练模型转化为paddle模型格式，具体可以看transfermodeltorch2paddle.py，里面的参数需要根据具体情况进行修改

## 2、训练脚本
**从Kinetics预训练的模型开始训练Flow流可以使用以下命令：**
``` bash
python Flow_train.py --dataset HMDB51 --modality Flow --n_classes 400 --n_finetune_classes 51 --batch_size 64 \
    --checkpoint 1 --sample_duration 16 --model resnext --model_depth 101 --frame_dir "dataset/hmdb_flowframe" \
    --annotation_path "dataset/hmdb51_split01" --result_path "results/" --Flow_premodel_path "models/Flow_Kinetics_16f" \
    --Flow_resume_path "models/model_Flow" --ft_begin_index 4 --split 1
```
注意：上面的训练脚本在使用过程中数据路径及模型路径要修改成自己的，frame_dir指提取的视频帧及光流图像的路径、annotation_path指图像标签的路径（split01），Flow_premodel_path指预训练模型的路径，Flow_resume_path指模型的保存路径
   
**从Kinetics预训练的模型开始训练MARS流可以使用以下命令：**
```bash
python MARS_train.py --dataset HMDB51 --modality RGB_Flow --n_classes 400  --n_finetune_classes 51     \
    --batch_size 64 --checkpoint 1 --sample_duration 16 --model resnext --model_depth 101      \
    --frame_dir "dataset/hmdb_flowframe" --annotation_path "dataset/hmdb51_split01" --result_path "results/" \
    --MARS_premodel_path "models/MARS_Kinetics_16f" --MARS_resume_path "models/model_MARS" \
    --Flow_resume_path "models/model_Flow/model_Flow_XX" --output "avgpool" --ft_begin_index 4  --split 1 \
    --MARS_alpha 50 
```
注意：上面的训练脚本在使用过程中数据路径及模型路径要修改成自己的，frame_dir指提取的视频帧及光流图像的路径、annotation_path指图像标签的路径（split01），MARS_premodel_path指预训练模型的路径，MARS_resume_path指模型的保存路径，Flow_resume_path是训练好的Flow流模型的路径

**从Kinetics预训练的模型开始训练RGB流可以使用以下命令：**
```bash
python RGB_train.py --dataset HMDB51 --modality RGB --n_classes 400 --n_finetune_classes 51 --batch_size 64 \
    --checkpoint 1 --sample_duration 16 --model resnext --model_depth 101 --frame_dir "dataset/hmdb_flowframe" \
    --annotation_path "dataset/hmdb51_split01" --result_path "results/" --RGB_premodel_path "models/RGB_Kinetics_16f" \
    --RGB_resume_path "models/model_RGB" --ft_begin_index 4 --split 1
```
注意：上面的训练脚本在使用过程中数据路径及模型路径要修改成自己的，frame_dir指提取的视频帧及光流图像的路径、annotation_path指图像标签的路径（split01），RGB_premodel_path指预训练模型的路径，RGB_resume_path指模型的保存路径
## 3、模型精度
Model|Top1
---|---
RGB|0.681
FLOW|0.719
MARS|0.721

## 4、测试脚本
**测试Flow流模型可以使用以下命令：**
```bash
python test_single_stream.py --dataset HMDB51 --modality Flow --n_classes 51 --batch_size 1  --checkpoint 1 \
--sample_duration 16 --model resnext --model_depth 101 --result_path "results/" --frame_dir "dataset/hmdb_flowframe"  \
--annotation_path "dataset/hmdb51_split01"   --Flow_resume_path "models/model_Flow/model_Flow_XX"  --split 1
```
**测试MARS流模型可以使用以下命令：**
```bash
python test_single_stream.py --dataset HMDB51 --modality RGB_Flow --n_classes 51 --batch_size 1  --checkpoint 1 \
    --sample_duration 16 --model resnext --model_depth 101 --result_path "results/" --frame_dir "dataset/hmdb_flowframe"  \
    --annotation_path "dataset/hmdb51_split01" --MARS_resume_path "models/model_MARS/model_MARS_XX"  --split 1
```
**测试RGB流模型可以使用以下命令：**
```bash
python test_single_stream.py --dataset HMDB51 --modality RGB --n_classes 51 --batch_size 1  --checkpoint 1 \
    --sample_duration 16 --model resnext --model_depth 101 --result_path "results/" --frame_dir "dataset/hmdb_flowframe"  \
    --annotation_path "dataset/hmdb51_split01"   --RGB_resume_path "models/model_Flow/model—_RGB_XX"  --split 1
```
注意：需要修改Flow_resume_path、MARS_resume_path、RGB_resume_path到训练好的模型路径
