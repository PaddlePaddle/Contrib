## multigpu
python main.py --decay=1e-4 --lr=0.001 --layer=3 --multigpu=True --dataset="gowalla" --topks="[20]" --recdim=64
## single gpu
python main.py --decay=1e-4 --lr=0.001 --layer=3 --dataset="gowalla" --topks="[20]" --recdim=64
## multicpu
python main.py --decay=1e-4 --lr=0.001 --layer=3 --multicpu=True --dataset="gowalla" --topks="[20]" --recdim=64