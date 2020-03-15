# Spatiotemporal Model of COVID-19 Spread

This repository contains two spatiotemporal prediction models based on PaddlePaddle and Paddle Graph Learning (PGL). The two models could be used to predict the COVID-19 outbreak (the estimates of infected patients per day) in major cities of mainland China.

本仓库中有两个基于PaddlePaddle以及Paddle Graph Learning (PGL) (https://github.com/PaddlePaddle/PGL) 时空序列预测模型。这两个模型被用于2019新冠疫情在中国大陆主要城市传播 (每日新增人数估计) 的预测。

## Data Specification

While the program leverages human mobility data collected from the Baidu Maps as the input, we here only provide mock data as reference. Please see also in the Baidu Open Mobility Data (https://qianxi.baidu.com/?from=mappc) or contact Baidu Data Federation Platform for the data usage. More details about the models have been specified in the two directories.

模型数据来自于百度地图迁徙数据，在本项目中我们仅提供mock_data，请自行使用百度地图公开的迁徙数据 (https://qianxi.baidu.com/?from=mappc) 或者联系使用百度数据联邦平台。模型的输入输出在两个模型各自的目录中有详细描述。



## Acknowledgments and Disclaimer

The human mobility data for training and prediction is provided by Baidu Map during development and test. The deep learning functionalities were implemented using PaddlePaddle and Paddle Graph Learning (PGL) frameworks. All the training and prediction processes were carried out using Baidu Data Federation Platform on top of Baidu AI Cloud. The models are independently designed by Xiamen university. Baidu is not responsible to the prediction results as well as any consequences related. Please contact Baidu Data Federation Platform via lucliuji (WeChat), when the human mobility data is needed.

该模型训练与预测所需之用户移动数据由百度地图提供。深度学习代码基于由百度飞桨 (https://github.com/PaddlePaddle/) 以及飞桨图学习模型(https://github.com/PaddlePaddle/PGL) 实现。所有训练与预测过程通过百度数邦平台数据提供、并由百度智能云算力支撑完成。该模型设计由厦门大学独立提供，百度对该模型的预测结果不持立场，对使用该模型预测所造成的任何后果不承担责任。如需要使用百度移动数据请联系百度数邦平台：lucliuji（微信）
