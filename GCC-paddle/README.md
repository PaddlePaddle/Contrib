

### 团队成员： 小木桌，chf544062970， 江江123
### 复现结果要点
```
本实现是基于官方的 [pytorch版本](https://github.com/THUDM/GCC) 改编，最后完全对齐forward,backward 

我们用paddle实现了DGL的backend， 这样DGL库可以在paddle上运行 

我们的实现用到paddlepaddle v2.1.0 版本的register_hook来控制gradient flow，所以需要安装 v2.1.0 版本
```

1、使用的数据集、模型文件及完整的复现代码

- `数据集` ： 采用  hindex , 用原代码自动下载的， 在gcc-paddle/data/hindex
- `完整的复现代码` 在这个项目的folder：gcc-paddle (论文实现代码)，  paddorch (提供pytorch接口的paddle实现), [dgl](https://github.com/chfhf/dgl-paddle) (DGL库的paddle backend 实现)

关于我写的torch接口代码请参考 [pytorch 转 paddle 心得](https://blog.csdn.net/weixin_48733317/article/details/108176827)
有兴趣了解的朋友可以看我在这个[视频](https://aistudio.baidu.com/aistudio/education/lessonvideo/698277)的Paddorch介绍（10分钟位置开始），
之前我用paddorch库复现了3个GAN类别的项目。

值得注意的是虽然说这个是GCC的paddle版本，但你基本上看不到paddle api接口，因为都被我们在paddorch库中重新封装了， 所以代码看起来就跟torch一样 


2、提供具体详细的说明文档(或notebook)，内容包括:

(1) 数据准备与预处理步骤

- 数据集自动下载，没有其他预处理步骤


(2) 训练脚本/代码，最好包含训练一个epoch的运行日志

- 在下面的cells 包含pretraining step 和finetune step的所有训练记录，和所有训练的命令行（完整训练记录参考下面） 
- pretrain step 我们跑了10个epoch 
- finetune step, 我们按照论文设置一样做了10-fold cross validation，每一个fold，training了20个epoch

(3) 测试脚本/代码，必须包含评估得到最终精度的运行日志

- 原来的官方代码没有独立的测试脚本，测试是包含在train.py， 
- 我们单独写一个测试脚本 `python eval_model.py ` 输出10-fold CV的Accuracy数值,平均值和标准差




(4) 最终精度，如精度相比源码有提升，需要说明精度提升用到的方法与技巧(不可更换网络主体结构，不可将测试集用于训练)

#### H-Index的10-cross validation 的 Accuracy：77.18%， 验收要求 76.9%

注意的是，官方代码用了sklearn.metrics.f1_score，我们测试过sklearn.metrics.accuracy_score算出来的数值是完全一样的。
原因是binrary classification和正负样本完全balance的情况下，F1=Accuracy


(5) 其它学员觉得需要说明的地方
-  一定要用32G GPU ，非常占显存
-  安装paddlepaddle v2.1.0 和DGL , Paddorch, 具体安装脚本在dgl/install_aistudio.sh
-  我们这里实现了moco版本，没有测试过E2E版本

3、上传最终训练好的模型文件
- 在`gcc-paddle/models` 

4、如评估结果保存在json文件中，可上传最终评估得到的json文件
没有生成json文件， 但可以下载visualdl`gcc-paddle/tensorboard` 目录进行评价


 =============================================================================================


<p align="center">
  <img src="fig.png" width="500">
  <br />
  <br />
  <a href="https://github.com/THUDM/GCC/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/THUDM/GCC" /></a>
  <a href="https://github.com/ambv/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

-------------------------------------

# GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training

Original implementation for paper [GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training](https://arxiv.org/abs/2006.09963).

GCC is a **contrastive learning** framework that implements unsupervised structural graph representation pre-training and achieves state-of-the-art on 10 datasets on 3 graph mining tasks.

- [GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training](#gcc-graph-contrastive-coding-for-graph-neural-network-pre-training)
  - [Installation](#installation)
    - [Requirements](#requirements)
  - [Quick Start](#quick-start)
    - [Pretraining](#pretraining)
      - [Pre-training datasets](#pre-training-datasets)
      - [E2E](#e2e)
      - [MoCo](#moco)
      - [Download Pretrained Models](#download-pretrained-models)
    - [Downstream Tasks](#downstream-tasks)
      - [Downstream datasets](#downstream-datasets)
      - [Node Classification](#node-classification)
        - [Unsupervised (Table 2 freeze)](#unsupervised-table-2-freeze)
        - [Supervised (Table 2 full)](#supervised-table-2-full)
      - [Graph Classification](#graph-classification)
        - [Unsupervised (Table 3 freeze)](#unsupervised-table-3-freeze)
        - [Supervised (Table 3 full)](#supervised-table-3-full)
      - [Similarity Search (Table 4)](#similarity-search-table-4)
  - [❗ Common Issues](#-common-issues)
  - [Citing GCC](#citing-gcc)
  - [Acknowledgements](#acknowledgements)

## Installation

### Requirements

- Linux with Python ≥ 3.6
- [PyTorch ≥ 1.4.0](https://pytorch.org/)
- [0.5 > DGL ≥ 0.4.3](https://www.dgl.ai/pages/start.html)
- `pip install -r requirements.txt`
- Install [RDKit](https://www.rdkit.org/docs/Install.html) with `conda install -c conda-forge rdkit=2019.09.2`.

## Quick Start

<!--
## How to process data

```
python x2dgl.py --graph-dir data_bin/kdd17 --save-file data_bin/dgl/graphs.bin
```
-->

### Pretraining

#### Pre-training datasets

```bash
python scripts/download.py --url https://drive.google.com/open?id=1JCHm39rf7HAJSp-1755wa32ToHCn2Twz --path data --fname small.bin
# For regions where Google is not accessible, use
# python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/b37eed70207c468ba367/?dl=1 --path data --fname small.bin
```

#### E2E

Pretrain E2E with `K = 255`:

```bash
bash scripts/pretrain.sh <gpu> --batch-size 256
```

#### MoCo

Pretrain MoCo with `K = 16384; m = 0.999`:

```bash
bash scripts/pretrain.sh <gpu> --moco --nce-k 16384
```

#### Download Pretrained Models

Instead of pretraining from scratch, you can download our pretrained models.

```bash
python scripts/download.py --url https://drive.google.com/open?id=1lYW_idy9PwSdPEC7j9IH5I5Hc7Qv-22- --path saved --fname pretrained.tar.gz
# For regions where Google is not accessible, use
# python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/cabec37002a9446d9b20/?dl=1 --path saved --fname pretrained.tar.gz
```

### Downstream Tasks

#### Downstream datasets

```bash
python scripts/download.py --url https://drive.google.com/open?id=12kmPV3XjVufxbIVNx5BQr-CFM9SmaFvM --path data --fname downstream.tar.gz
# For regions where Google is not accessible, use
# python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/2535437e896c4b73b6bb/?dl=1 --path data --fname downstream.tar.gz
```

Generate embeddings on multiple datasets with

```bash
bash scripts/generate.sh <gpu> <load_path> <dataset_1> <dataset_2> ...
```

For example:

```bash
bash scripts/generate.sh 0 saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/current.pth usa_airport kdd imdb-binary
```

#### Node Classification

##### Unsupervised (Table 2 freeze)

Run baselines on multiple datasets with `bash scripts/node_classification/baseline.sh <hidden_size> <baseline:prone/graphwave> usa_airport h-index`.

Evaluate GCC on multiple datasets:

```bash
bash scripts/generate.sh <gpu> <load_path> usa_airport h-index
bash scripts/node_classification/ours.sh <load_path> <hidden_size> usa_airport h-index
```

##### Supervised (Table 2 full)

Finetune GCC on multiple datasets:

```bash
bash scripts/finetune.sh <load_path> <gpu> usa_airport
```

Note this finetunes the whole network and will take much longer than the freezed experiments above.

#### Graph Classification

##### Unsupervised (Table 3 freeze)

```bash
bash scripts/generate.sh <gpu> <load_path> imdb-binary imdb-multi collab rdt-b rdt-5k
bash scripts/graph_classification/ours.sh <load_path> <hidden_size> imdb-binary imdb-multi collab rdt-b rdt-5k
```

##### Supervised (Table 3 full)

```bash
bash scripts/finetune.sh <load_path> <gpu> imdb-binary
```

#### Similarity Search (Table 4)

Run baseline (graphwave) on multiple datasets with `bash scripts/similarity_search/baseline.sh <hidden_size> graphwave kdd_icdm sigir_cikm sigmod_icde`.

Run GCC:

```bash
bash scripts/generate.sh <gpu> <load_path> kdd icdm sigir cikm sigmod icde
bash scripts/similarity_search/ours.sh <load_path> <hidden_size> kdd_icdm sigir_cikm sigmod_icde
```

## ❗ Common Issues

<details>
<summary>
"XXX file not found" when running pretraining/downstream tasks.
</summary>
<br/>
Please make sure you've downloaded the pretraining dataset or downstream task datasets according to GETTING_STARTED.md.
</details>

<details>
<summary>
Server crashes/hangs after launching pretraining experiments.
</summary>
<br/>
In addition to GPU, our pretraining stage requires a lot of computation resources, including CPU and RAM. If this happens, it usually means the CPU/RAM is exhausted on your machine. You can decrease `--num-workers` (number of dataloaders using CPU) and `--num-copies` (number of datasets copies residing in RAM). With the lowest profile, try `--num-workers 1 --num-copies 1`.

If this still fails, please upgrade your machine :). In the meanwhile, you can still download our pretrained model and evaluate it on downstream tasks.
</details>

<details>
<summary>
Having difficulty installing RDKit.
</summary>
<br/>
See the P.S. section in [this](https://github.com/THUDM/GCC/issues/12#issue-752080014) post.
</details>

## Citing GCC

If you use GCC in your research or wish to refer to the baseline results, please use the following BibTeX.

```
@article{qiu2020gcc,
  title={GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training},
  author={Qiu, Jiezhong and Chen, Qibin and Dong, Yuxiao and Zhang, Jing and Yang, Hongxia and Ding, Ming and Wang, Kuansan and Tang, Jie},
  journal={arXiv preprint arXiv:2006.09963},
  year={2020}
}
```

## Acknowledgements

Part of this code is inspired by Yonglong Tian et al.'s [CMC: Contrastive Multiview Coding](https://github.com/HobbitLong/CMC).
