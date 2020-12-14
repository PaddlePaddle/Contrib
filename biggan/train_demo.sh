##training
cd  BigGAN-paddle/; python train.py  --shuffle --batch_size 512   --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800    --num_D_steps 4 \
	 --G_lr 2e-3 --D_lr 2e-3  --dataset C10  --G_ortho 0.0  --G_attn 0 --D_attn 0  --G_init N02 --pbar tqdm \
  --D_init $N02  --ema --use_ema --ema_start 1000  --test_every 800 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 

cd ..
