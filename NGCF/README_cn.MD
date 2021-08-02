## NGCF-paddle 简介
NGCF-paddle 是SIGIR 2019 论文  Neural Graph Collaborative Filtering.
以paddle 2.0.1版本为基础借助pgl实现。
在这项工作中，作者将图与传统协同过滤算法结合为NGCF，取得较好的结果
## 环境
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
## datasets
数据格式为
[userid1 itemid1 itemid2 ......]
[userid2 itemid7 itemid0 ......]
[userid3 itemid12 itemid22 ......]
[userid4 itemid11 itemid92 ......]