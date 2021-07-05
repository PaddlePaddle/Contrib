# Deep Match to Rank

- Update: 优化了数据集中Price的归一化方式，最优学习率更新为0.008（未尝试更小的学习率，有可能0.006~0.0075会取得更好结果），最优AUC更新为0.6441。

## 最优结果
- AUC: 0.6441
- RI: 0.74%

## 最优参数
- lr：0.008
- batch_size: 5120
- 优化器：Adam


## 准备开发环境
- 下载PaddleRec
- [数据集清洗方法](https://aistudio.baidu.com/aistudio/projectdetail/1805731)
- [原始数据集](https://aistudio.baidu.com/aistudio/datasetdetail/79462)
- [清洗后数据集](https://aistudio.baidu.com/aistudio/datasetdetail/81892)

```
==> raw_sample.csv <== nonclk,clk对应alimama_sampled.txt最后一列（266），点击与否。用前面7天的做训练样本（20170506-20170512），用第8天的做测试样本（20170513），time_stamp 1494032110 stands for 2017-05-06 08:55:10。pid要编码为类别数字。
user,time_stamp,adgroup_id,pid,nonclk,clk
581738,1494137644,1,430548_1007,1,0

==> behavior_log.csv <== 对应alimama_sampled.txt中[0:150]列（列号从0开始），需要根据raw_sample.csv每行记录查找对应的50条历史数据，btag要编码为类别数字
user,time_stamp,btag,cate,brand
558157,1493741625,pv,6250,91286

==> user_profile.csv <== 对应alimama_sampled.txt中[250:259]列（列号从0开始）
userid,cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level 
234,0,5,2,5,,3,0,3

==> ad_feature.csv <== 对应alimama_sampled.txt中[259:264]列（列号从0开始）,price需要标准化到0~1
adgroup_id,cate_id,campaign_id,customer,brand,price
63133,6406,83237,1,95471,170.0
```


## 测试模型


```python
# 准备数据
os.makedirs(os.path.join(data_base_dir, 'data/sample_data/train'), exist_ok=True)
os.makedirs(os.path.join(data_base_dir, 'data/sample_data/test'), exist_ok=True)

i = 0
f_train = open(os.path.join(data_base_dir, 'data/sample_data/train/alimama_sampled_train.txt'), 'w')
f_test = open(os.path.join(data_base_dir, 'data/sample_data/test/alimama_sampled_test.txt'), 'w')
with open(os.path.join(data_base_dir, 'data/alimama_sampled.txt'), 'r') as f:
    line = f.readline()
    while line:
        if i % 10 < 2:
            f_test.write(line)
        else:
            if line.strip()[-1] == '1':
                up_cnt = 15  # 15, epoch 2, train 0.91, test 0.552003; 20, epoch 2, train 0.90, test 0.53; 5, epoch 2, train 0.89, test 0.52; 10, epoch 2, train 0.90, test 0.507
            else:
                up_cnt = 1
            for up_j in range(up_cnt):
                f_train.write(line)

        i += 1
        line = f.readline()

print(f'total lines: {i}')
f_train.close()
f_test.close()

# !head -n 1 data/sample_data/train/alimama_sampled_train.txt
# !head -n 1 data/sample_data/test/alimama_sampled_test.txt

# !rm -r data/sample_data/train/.ipynb_checkpoints
# !rm -r data/sample_data/test/.ipynb_checkpoints
```


```python
# # 动态图训练
# !python -u ../../../tools/trainer.py -m config.yaml
```


```python
# # 动态图预测
# !python -u ../../../tools/infer.py -m config.yaml
```

## 全量数据训练


```python
# %cd ~/work/PaddleRec/models/rank/dmr/

import os
if not os.path.isdir(os.path.join(data_base_dir, 'data/full_data/train')):
    !unzip -o -d /home/aistudio/data/ /home/aistudio/data/data81892/dataset_full.zip
    os.makedirs(os.path.join(data_base_dir, 'data/full_data/train'), exist_ok=True)
    os.makedirs(os.path.join(data_base_dir, 'data/full_data/test'), exist_ok=True)
    !mv /home/aistudio/data/work/train_sorted.csv /home/aistudio/data/full_data/train/
    !mv /home/aistudio/data/work/test.csv /home/aistudio/data/full_data/test/
```


```python
# 动态图训练
print('Start Training...')

!python -u ../../../tools/trainer.py -m config_bigdata.yaml

!rm -rf ../../../tools/utils/__pycache__  __pycache__

print('End Training...')
```

## 模型结果


```python
# 动态图预测
print('Start Testing...')

!python -u ../../../tools/infer.py -m config_bigdata.yaml

!rm -rf ../../../tools/utils/__pycache__  __pycache__

print('End Testing...')
```

## 清理大文件


```python
# ! rm -r ~/work/PaddleRec/models/rank/dmr/output_model_dmr/  ~/work/PaddleRec/models/rank/dmr/output_model_dmr_fulldata/  ~/work/PaddleRec/models/rank/dmr/visualDL_log/ ~/work/PaddleRec/models/rank/dmr/data/
```
