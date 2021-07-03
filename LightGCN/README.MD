## LightGCN-paddle introduction
This is the paddle2.0.1 implementation for our SIGIR 2020 paper:
Simplifying and Powering Graph Convolution Network for Recommendation 
In this work, authors aim to simplify the design of GCN to make it more concise and appropriate for recommendation. 
authors propose a new model named LightGCN,
including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering
## Enviroment Requirement 
pip install -r requirements.txt
## quick start
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64
## results
gowalla:
	Recall	 ndcg	precision
layer=3	0.1824	0.1547	0.05589
amazon-book:
        Recall	 ndcg	precision
layer=3	0.04086  0.0314	0.01683
yelp2018:
        Recall	 ndcg	precision
layer=3	0.0632	0.0519	0.0285
## the formats of datasets
[userid1 itemid1 itemid2 ......]
[userid2 itemid7 itemid0 ......]
[userid3 itemid12 itemid22 ......]
[userid4 itemid11 itemid92 ......]
## multi-gpu
please reference paddlepaddle，https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/cluster_quick_start_cn.html