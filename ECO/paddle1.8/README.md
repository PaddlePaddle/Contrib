# PaddlePaddle implementation of ECO

["ECO: Efficient Convolutional Network for Online Video Understanding, European Conference on Computer Vision (ECCV), 2018." By Mohammadreza Zolfaghari, Kamaljeet Singh, Thomas Brox](https://arxiv.org/abs/1804.09066)

[The author's Repository for ECO](https://github.com/mzolfaghari/ECO-efficient-video-understanding)


## Introduction

This repository contains all the required codes and results for the [PaddlePaddle](https://github.com/paddlepaddle) implementation of the paper ECO: Efficient Convolutional Network for Online Video Understanding.

## Environment

Python 3.7.4

PaddlePaddle 1.8.0

## Repository Structure

* data: the dataset for training and testing, downloading by yourself is needed
* configs: model, training and testing configuration parameters
* model: the eco-full model
* trained_model: the trained model
* training_results: the images for the training analysis
* result_data: the training information saved as npz, accuray vs. steps
* avi2jpg.py: for converting avi files to jpg files
* jpg2pkl.py: for converting jpg files to avi files
* data_list_gener.py: for generating training list and testing list based on Split01
* reader.py: for sampling frames for training, evaluating and testing
* config.py: project configuration file
* utils.py: project utils file
* train.py: for training
* test.py: for testing
* requirements.txt: the required packages
* README.md: repository description file


## Training on manually splited UCF-101

The UCF-101 dataset is manually splited between training, evaluating and testing with 9090 videos for training, 909 videos for evaluating and 3321 videos for testing. 

The final accuracy result on the testing part is about 94.34%.

The later fine-tuning stage for training and evaluating is shown in the following image.

![training accuracy](https://github.com/eepgxxy/ECO_PaddlePaddle/blob/master/training_results/train_1.png)

![eval accuracy](https://github.com/eepgxxy/ECO_PaddlePaddle/blob/master/training_results/eval_1.png)

## Training on trainlist01 of UCF-101

Preliminary training is also done on the trainlist01 of UCF-101 dataset and testing is done on testlist01. 

In order to use the pretrained model, we created a new label file myclassID.txt.

The final accuracy result on the testing part is currently about 97.568% for seg_num 12, 97.938% for seg_num 24, 97.224% for seg_num 32.

Train and eval stages for seg_num 24 and seg_num 32 are illustrated by the following images:

![training and eval for seg_num 24](https://github.com/eepgxxy/ECO_PaddlePaddle/blob/master/training_results/train_eval_24.png)

![training and eval for seg_num 32](https://github.com/eepgxxy/ECO_PaddlePaddle/blob/master/training_results/train_eval_32.png)

## Other experiments

The network architecture for this model is Inception + 3DResNet based on the original paper. Other network architectures are explored, such as ResNet + 3DResNet, and similar results are found.

## TBD

Further experiments will be done and update will be made here.

