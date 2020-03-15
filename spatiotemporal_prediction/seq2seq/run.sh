#!/bin/bash 
python data_process.py 
python train.py --use_cuda --epoch 220 
python test.py --use_cuda
