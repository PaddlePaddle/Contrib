#!/bin/bash

python sample.py --shuffle --batch_size 128 --G_batch_size 256  \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4   \
--dataset C10 \
--load_weights currentbest \
--G_ortho 0.0  --G_eval_mode  --sample_sheets --sample_interps  --sample_random  --sample_inception_metrics \
--G_attn 0 --D_attn 0  --sample_trunc_curves 0.2_0.1_1.0 \
--G_init N02 --D_init N02  \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0
