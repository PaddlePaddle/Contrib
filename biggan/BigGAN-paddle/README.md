# BigGAN-Paddle
This is replementation of BigGAN using DL framework [PaddlePaddle](https://github.com/PaddlePaddle/Paddle).

 

This code is tested in Cifar10 data, and in only single GPU mode

This repo is modified based on original BigGAN [pytorch version](https://github.com/ajbrock/BigGAN-PyTorch).

## How To Use This Code
You will need:

- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle), version 1.8.4
- [Paddorch](https://github.com/zzz2010/paddle_torch)
- tqdm, numpy, scipy, and h5py
- The Cifar10 training set
 
 

### run training
```sh
sh scripts/launch_cifar_ema.sh
```

### run evaluation
```sh
sh scripts/sample_cifar_ema.sh  
```
## Pretrained models
I include two pretrained model checkpoints (with G, D, the EMA copy of G, and the state dict) and inceptionV3 model for FID/inception score evalution:

[cifar10 BigGAN model](https://aistudio.baidu.com/aistudio/datasetdetail/52466)
