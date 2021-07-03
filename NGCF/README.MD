## NGCF-paddle introduction
This is the paddle2.0.1 implementation for SIGIR 2019 paper:
Neural Graph Collaborative Filtering.
In this work, authors aim to collaborative filtering with graph structrue.
## Enviroment Requirement 
pip install -r requirements.txt
## quick start
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64
## results
gowalla:
	    Recall	 ndcg
layer=3	0.1575	0.13343
amazon-book:
        Recall	 ndcg
layer=3	0.0339  0.0261
## the formats of datasets
[userid1 itemid1 itemid2 ......]
[userid2 itemid7 itemid0 ......]
[userid3 itemid12 itemid22 ......]
[userid4 itemid11 itemid92 ......]