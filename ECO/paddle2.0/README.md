# Paddle2.0 implementation of ECO

["ECO: Efficient Convolutional Network for Online Video Understanding, European Conference on Computer Vision (ECCV), 2018." By Mohammadreza Zolfaghari, Kamaljeet Singh, Thomas Brox](https://arxiv.org/abs/1804.09066)

[The author's Repository for ECO](https://github.com/mzolfaghari/ECO-efficient-video-understanding)


## Introduction

This repository contains all the required codes and results for the [Paddle2.0](https://github.com/paddlepaddle) implementation of the paper ECO: Efficient Convolutional Network for Online Video Understanding. 

## Environment

Python 3.7.4

PaddlePaddle 2.0.0-rc1

## Repository Structure

* configs: model, training and testing configuration parameters
* model: the eco-full model
* result: the training information saved as npz, accuray vs. epochs
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


## Training on splited UCF-101 with trainlist01.txt

The UCF-101 dataset is trained using trainlist01.txt and tested using testlist01.txt. 

The final accuracy result on the testing part is about 94.18% without too much finetuning.





