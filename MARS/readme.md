## 数据下载及准备工作
 数据集：数据可使用HMDB51，可以在https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/下载HMDB51数据集

 **准备工作**
1. 安装带GPU支持的opencv
2. 从视频中提取帧可以使用
```bash
python utils1/extract_frames.py path_to_video_files path_to_extracted_frames start_class end_class
```  
3. 从视频中提取帧+光流可以使用
```bash
export OPENCV=path_where_opencv_is_installed

g++ -std=c++11 tvl1_videoframes.cpp -o tvl1_videoframes -I${OPENCV}include/opencv4/ -L${OPENCV}lib64 -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudaarithm

python utils1/extract_frames_flows.py path_to_video_files path_to_extracted_flows_frames start_class end_cl
```
4. 需要从https://drive.google.com/drive/folders/1OVhBnZ_FmqMSj6gw9yyrxJJR8yRINb_G?usp=sharing下载预训练的pytorch模型

5. 将pytorch预训练模型转化为paddle模型格式，具体可以看transfermodeltorch2paddle.py里面的参数需要根据具体情况进行修改

## 训练Flow流可以使用以下命令：
``` bash
python Flow_train.py --dataset HMDB51 --modality Flow --n_classes 400  --n_finetune_classes 51 \
    --batch_size 64 --log 1 --checkpoint 5 --sample_duration 16 --model resnext --model_depth 101 \
    --frame_dir "dataset/hmdb_flowframe" --annotation_path "dataset/hmdb51_label" --result_path "results/" \
    --Flow_premodel_path "models/Flow" --Flow_resume_path "models/model_Flow"
```
     
训练RGB流和训练MARS流的命令可以仿照上面训练FLOW流的命令
     
## 测试flow流可以使用以下命令：
```bash
 python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
      --dataset HMDB51 --modality Flow --sample_duration 16  \
      --Flow_resume_path  Your model saved path \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --result_path "results/"
```
测试RGB流和训练MARS流的命令可以仿照上面测试FLOW流的命令

## 模型精度
Model|Top1
---|---
RGB|0.681
FLOW|0.719
MARS|0.721
