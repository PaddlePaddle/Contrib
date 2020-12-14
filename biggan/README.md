# BigGAN-Paddle
This is replementation of BigGAN using DL framework [PaddlePaddle](https://github.com/PaddlePaddle/Paddle).

 

This code is tested in Cifar10 data, and in only single GPU mode

This repo is modified based on original BigGAN [pytorch version](https://github.com/ajbrock/BigGAN-PyTorch).

## How To Use This Code
You will need:

- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle), version 1.8.4
- [Paddorch](https://github.com/zzz2010/paddle_torch), a version is provided in this repo
- tqdm, numpy, scipy, and h5py
- CuDNN 7, need to match paddlepaddle version
- to run the demo: The Cifar10 training set, inception pretrained model to compute FID and IS score


 
### Download Cifar10 Dataset and  inception pretrained model
Both dataset and pretrained model could be downloaded from AIstudio
- [Cifar10](https://aistudio.baidu.com/aistudio/datasetdetail/39555)
- [Inception pretrained model](https://aistudio.baidu.com/aistudio/datasetdetail/52466)


Use below script will download the data and exact to the correct directory.
```sh
sh download.sh
```
 
### Install Dependence
use below script to install paddorch and other dependence, except CuDNN
```shell script
sh install_dependence.sh
```

### run training demo
```sh
sh train_demo.sh
```

### run evaluation demo
```sh
sh evaluation_demo.sh
```

