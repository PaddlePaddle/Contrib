# PF-Net_PaddlePaddle
Author: CAO RUI

PaddlePaddle Implementation for [PF-Net: Point Fractal Network for 3D Point Cloud Completion](https://arxiv.org/abs/2003.00410)

# Prerequisites

Python: Python 3.6+

[PaddlePaddle v1.8.3](https://github.com/PaddlePaddle/Paddle) : Deep learning framework

[open3d](http://www.open3d.org/docs/release/): 3D vision visualization toolbox

# Install

Clone the repo

```git clone https://github.com/63445538/PF-Net_PaddlePaddle.git```

Then, run script```download_shapenet_part16_catagories.sh``` in folder ```dataset``` for preparing shapenet part dataset.

Download pretrained weight for test:[gdrive](https://drive.google.com/file/d/1Ecij0Th7573Xvp8d98GhJn3DNCe2mzJ6/view?usp=sharing), [baidu pan](https://pan.baidu.com/s/1pMgq_kBYQ73nIrU5jUTK6Q)(password:wwtx), and copy it into folder ```Checkpoints```.

# Train

```python Train_FPNet.py```

# Test

```python Test_FPNet.py```

# Result

Here show the completion results:
![image](https://github.com/63445538/PF-Net_PaddlePaddle/blob/master/img/1.png)
![image](https://github.com/63445538/PF-Net_PaddlePaddle/blob/master/img/2.png)
![image](https://github.com/63445538/PF-Net_PaddlePaddle/blob/master/img/3.png)

The following figure shows point cloud completion results of the missing point cloud
![image](https://github.com/63445538/PF-Net_PaddlePaddle/blob/master/img/result1.png)

# Acknowledgement

[PF-Net-Point-Fractal-Network](https://github.com/zztianzz/PF-Net-Point-Fractal-Network) : Offical PyTorch implementation.
