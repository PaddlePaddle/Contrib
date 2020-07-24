1、需要从https://drive.google.com/drive/folders/1OVhBnZ_FmqMSj6gw9yyrxJJR8yRINb_G?usp=sharing下载预训练的pytorch模型

2、将pytorch预训练模型转化为paddle模型格式
     具体可以看transfermodeltorch2paddle.py
     里面的参数需要根据具体情况进行修改

3、训练Flow流可以使用以下命令：
python Flow_train.py --dataset HMDB51 --modality Flow --n_classes 400  --n_finetune_classes 51 \
    --batch_size 64 --log 1 --checkpoint 5 --sample_duration 16 --model resnext --model_depth 101 \
    --frame_dir "dataset/hmdb_flowframe" --annotation_path "dataset/hmdb51_label" --result_path "results/" \
    --Flow_premodel_path "models/Flow" --Flow_resume_path "models/model_Flow"
训练RGB流和训练MARS流的命令可以仿照上面训练FLOW流的命令

