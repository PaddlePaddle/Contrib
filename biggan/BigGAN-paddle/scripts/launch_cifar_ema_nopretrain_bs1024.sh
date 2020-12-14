#!/bin/bash
#!/bin/bash
python train.py  --shuffle --batch_size 1024   --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 5000    --num_D_steps 4 \
 --G_lr 2e-4 --D_lr 2e-4  --dataset C10  --G_ortho 0.0  --G_attn 0 --D_attn 0  --G_init N02\
 --D_init N02  --ema --use_ema --ema_start 100  --test_every 500 --save_every 200 --num_best_copies 5 --num_save_copies 2 --seed 0    --resume 
