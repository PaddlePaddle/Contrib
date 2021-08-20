# yolactpaddle
* [**项目Aistudio链接**](https://aistudio.baidu.com/aistudio/projectdetail/525713)

* [**YOLACT++: Better Real-time Instance Segmentation**](https://arxiv.org/abs/1912.06218)

```
@misc{yolact-plus-arxiv2019,
  title         = {YOLACT++: Better Real-time Instance Segmentation},
  author        = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  year          = {2019},
  eprint        = {1912.06218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

本项目是YOLACT-550++的PaddlePaddle实现, 包含模型训练, 测试, 数据集等内容。
项目主体基于[**PaddleDetection V0.2**](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.2)


## 依赖库

- Python>=3.0
- opencv3
- numpy
- paddle>=1.7
- pycocotools

## 数据集

本项目中使用COCO2017数据集进行训练和评测，可以使用如下两种方式下载，数据集需要压缩至dataset/coco/目录下。
* [**Aistudio数据集**](https://aistudio.baidu.com/aistudio/datasetdetail/7122)
* [**参考脚本**](https://github.com/PaddlePaddle/models/blob/v1.5/PaddleCV/rcnn/dataset/coco/download.sh)


## 配置文件

- [网络结构配置文件](configs/yolactplus_r50_fpn.yml)
- [数据处理配置文件](configs/yolactplus_reader.yml)

## 模型下载
下载如下链接模型压缩至output/yolactplus_r50_fpn/下

```
https://aistudio.baidu.com/aistudio/datasetdetail/38635
```

## 训练

```
从头开始训练
python tools/train.py -c configs/yolactplus_r50_fpn.yml 
恢复断点，其中1020000为模型参数
python tools/train.py -c configs/yolactplus_r50_fpn.yml -r output/yolactplus_r50_fpn/1020000 
边训练边测试
python tools/train.py -c configs/yolactplus_r50_fpn.yml --eval
```

## 评估
```
python tools/eval.py -c configs/yolactplus_r50_fpn.yml
```

## 模型预测
```
python -u tools/infer.py -c configs/yolactplus_r50_fpn.yml \
                    --infer_img=demo/000000570688.jpg \
                    --output_dir=infer_output/ \
                    --draw_threshold=0.5 \
                    -o weights=output/yolactplus_r50_fpn/1020000  \
```

## 数据增强
原repo在数据预处理部分采用和ssd相同的方式，复现过程中发现RandomDistort效果不佳，便注释了该部分。
```
  # - !RandomDistort
  #   brightness_lower: 0.875
  #   brightness_upper: 1.125
  #   is_order: true
  - !ExpandImage
    max_ratio: 4
    prob: 0.5
    segm_flag: true
  - !RandomCrop
    allow_no_crop: true
```




