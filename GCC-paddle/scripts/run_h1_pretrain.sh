#!/bin/bash
python  train.py --moco --nce-k 16384 --num-workers 1 --num-copies 1 --alpha 0.999 --dataset h-index --gpu -1 --epochs 20
