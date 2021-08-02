## LightGCN-paddle 简介
LightGCN-paddle 是SIGIR 2020 论文 LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation 
以paddle 2.0.1版本为基础借助pgl实现。
在这项工作中，作者简化了GCN的设计，使其更简洁，更适合推荐。提出了一种新的协同过滤模型LightGCN，该模型只包含GCN邻域聚合中最基本的部分。
## 环境
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
## datasets
数据格式为
[userid1 itemid1 itemid2 ......]
[userid2 itemid7 itemid0 ......]
[userid3 itemid12 itemid22 ......]
[userid4 itemid11 itemid92 ......]
## 分布式训练
请参考paddlepaddle，https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/cluster_quick_start_cn.html