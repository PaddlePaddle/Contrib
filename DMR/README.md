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


```python
# %cd ~
import os
!ls /home/aistudio/data/
!ls work/
!python --version
!pip list | grep paddlepaddle
if not os.path.isdir('work/PaddleRec'):
    !cd work && git clone https://gitee.com/paddlepaddle/PaddleRec.git
```


```python
# !cd PaddleRec/ && unzip -o PaddleRec.zip
```


```python
os.makedirs('data/sample_data', exist_ok=True)

data_base_dir = '/home/aistudio'

if not os.path.exists('work/alimama_sampled.txt'):
    !cd work && wget https://raw.githubusercontent.com/lvze92/DMR/master/alimama_sampled.txt

if not os.path.exists(os.path.join(data_base_dir, 'data/alimama_sampled.txt')):
    !cp work/alimama_sampled.txt data/

# if not os.path.exists('PaddleRec/models/rank/dmr/data/sample_data/alimama_sampled.txt'):
#     os.makedirs('PaddleRec/models/rank/dmr/data/sample_data/', exist_ok=True)
#     !cp data/alimama_sampled.txt PaddleRec/models/rank/dmr/data/sample_data/
```

## 自定义Reader

参照models/rank/dnn 目录下的criteo_reader.py的实现方式

### 修改xx_reader.py
用户只需要修改class RecDataset中的__iter__函数, 通过python自带的yield方式输出每条数据，目前推荐使用numpy格式输出。

以line1为例 根据自定义函数, 实现对4个特征域的分别输出, yield的格式支持list。

yield [numpy.array([1]), numpy.array([2, 3]), numpy.array([100]), numpy.array([2.1,5.8,8.9])]

- Tips1: 目前的class必须命名为RecDataset, 用户只需要修改__iter__函数

- Tips2: 调试过程中可以直接print, 快速调研

### 修改config.yaml
详细的yaml格式可以参考进阶教程的yaml文档

yaml中的runner.train_reader_path 为训练阶段的reader路径

- Tips: importlib格式, 如test_reader.py，则写为train_reader_path: "test_reader"


```python
# 进入work目录
%cd work

current_wd = !pwd
if 'PaddleRec/models/rank/dmr' not in current_wd[0]:
    %cd 'PaddleRec/models/rank/dmr'
else:
    print(current_wd)
```


```python
%%writefile alimama_reader.py
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import numpy as np

from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list

    def __iter__(self):
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    l = l.strip().split(",")
                    l = ['0' if i=='' or i.upper()=='NULL' else i for i in l]  # handle missing values
                    output_list = np.array(l, dtype='float32')
                    yield output_list
```

## 自定义模型
### 动态图模型

Tips1: 必须在模型目录实现dygraph_model.py中的class DygraphModel，不能更改py文件名也不能更改class类名。

Tips2: 必须实现方法create_model, create_optimizer, create_metrics, train_forward, infer_forward。

Tips3: create_feeds和create_loss由train_forward和infer_forward内部调用，可以自定义方法名称。

#### create_model

返回模型的class, 一般是调用net.py中定义的组网。

#### create_feeds

解析batch_data, 返回paddle的tensor格式，在dataloader中yield是一条数据，注意这里返回的是Batch数据。

Tips: 因为动态图不需要占位符data, 这里实际返回的就是模型的输入tensor。

#### create_loss

由于采用了动静一致的设计理念和方便计算指标的独立，将loss部分单独抽出来实现在这个函数中，也可以直接在train_forward中定义loss部分

#### create_optimizer

定义优化器, 这里由用户自定义优化器。

#### create_metrics

定义评估指标，返回打印的key值和声明的指标

Tips: 返回的指标必须是paddle.metric中的指标

#### train_forward

自定义训练阶段，一般包含数据读入，计算loss损失，更新指标

Tips: 返回3个值，第一个必须是loss, 第二个是metric_list，可以为空list。第三个是想间隔打印的tensor dict, 可以返回None。

#### infer_forward

除了不返回loss之外其他和train_forward相同，支持和train阶段不同的组网。

```
query_prelu: [256, 50, 64]
dnn_layer1_prelu: [256, 50, 32]
query_prelu2: [256, 50, 64]
dnn0_prelu: [256, 512]
dnn1_prelu: [256, 256]
dnn2_prelu: [256, 128]
dnn3_prelu: [256, 1]
```


```python
%%writefile net.py
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np


class DMRLayer(nn.Layer):
    def __init__(self, user_size, lr, global_step,
                 cms_segid_size, cms_group_id_size, final_gender_code_size,
                 age_level_size, pvalue_level_size, shopping_level_size,
                 occupation_size, new_user_class_level_size, adgroup_id_size,
                 cate_size, campaign_id_size, customer_size,
                 brand_size, btag_size, pid_size,
                 main_embedding_size, other_embedding_size):
        super(DMRLayer, self).__init__()

        self.user_size = user_size
        self.lr = lr
        self.global_step = global_step
        self.cms_segid_size = cms_segid_size
        self.cms_group_id_size = cms_group_id_size
        self.final_gender_code_size = final_gender_code_size
        self.age_level_size = age_level_size
        self.pvalue_level_size = pvalue_level_size
        self.shopping_level_size = shopping_level_size
        self.occupation_size = occupation_size
        self.new_user_class_level_size = new_user_class_level_size
        self.adgroup_id_size = adgroup_id_size
        self.cate_size = cate_size
        self.campaign_id_size = campaign_id_size
        self.customer_size = customer_size
        self.brand_size = brand_size
        self.btag_size = btag_size
        self.pid_size = pid_size
        self.main_embedding_size = main_embedding_size
        self.other_embedding_size = other_embedding_size

        self.uid_embeddings_var = paddle.nn.Embedding(
            self.user_size,
            self.main_embedding_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="UidSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.mid_embeddings_var = paddle.nn.Embedding(
            self.adgroup_id_size,
            self.main_embedding_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="MidSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.cat_embeddings_var = paddle.nn.Embedding(
            self.cate_size,
            self.main_embedding_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="CatSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.brand_embeddings_var = paddle.nn.Embedding(
            self.brand_size,
            self.main_embedding_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="BrandSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.btag_embeddings_var = paddle.nn.Embedding(
            self.btag_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="BtagSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.dm_btag_embeddings_var = paddle.nn.Embedding(
            self.btag_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="DmBtagSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.campaign_id_embeddings_var = paddle.nn.Embedding(
            self.campaign_id_size,
            self.main_embedding_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="CampSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.customer_embeddings_var = paddle.nn.Embedding(
            self.customer_size,
            self.main_embedding_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="CustomSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.cms_segid_embeddings_var = paddle.nn.Embedding(
            self.cms_segid_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="CmsSegSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.cms_group_id_embeddings_var = paddle.nn.Embedding(
            self.cms_group_id_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="CmsGroupSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.final_gender_code_embeddings_var = paddle.nn.Embedding(
            self.final_gender_code_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="GenderSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.age_level_embeddings_var = paddle.nn.Embedding(
            self.age_level_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="AgeSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.pvalue_level_embeddings_var = paddle.nn.Embedding(
            self.pvalue_level_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="PvSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.shopping_level_embeddings_var = paddle.nn.Embedding(
            self.shopping_level_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="ShopSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.occupation_embeddings_var = paddle.nn.Embedding(
            self.occupation_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="OccupSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.new_user_class_level_embeddings_var = paddle.nn.Embedding(
            self.new_user_class_level_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="NewUserClsSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.pid_embeddings_var = paddle.nn.Embedding(
            self.pid_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="PidSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.position_his = paddle.to_tensor(np.arange(50))
        self.position_embeddings_var = paddle.nn.Embedding(
            50,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="PosSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.dm_position_his = paddle.to_tensor(np.arange(50))
        self.dm_position_embeddings_var = paddle.nn.Embedding(
            50,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="DmPosSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))
        self.query_layer = paddle.nn.Linear(16, 64, name='dm_align')
        self.query_prelu = paddle.nn.PReLU(num_parameters=50, init=0.1, name='dm_prelu')
        self.att_layer1_layer = paddle.nn.Linear(256, 80, name='dm_att_1')
        self.att_layer2_layer = paddle.nn.Linear(80, 40, name='dm_att_2')
        self.att_layer3_layer = paddle.nn.Linear(40, 1, name='dm_att_3')
        self.dnn_layer1_layer = paddle.nn.Linear(64, self.main_embedding_size, name='dm_fcn_1')
        self.dnn_layer1_prelu = paddle.nn.PReLU(num_parameters=50, init=0.1, name='dm_fcn_1')

        self.query_layer2 = paddle.nn.Linear(80, 64, name='dmr_align')
        self.query_prelu2 = paddle.nn.PReLU(num_parameters=50, init=0.1, name='dmr_prelu')
        self.att_layer1_layer2 = paddle.nn.Linear(256, 80, name='tg_att_1')
        self.att_layer2_layer2 = paddle.nn.Linear(80, 40, name='tg_att_2')
        self.att_layer3_layer2 = paddle.nn.Linear(40, 1, name='tg_att_3')

        self.logits_layer = paddle.nn.Linear(self.main_embedding_size, self.cate_size)

        def deep_match(item_his_eb, context_his_eb, mask, match_mask, mid_his_batch, EMBEDDING_DIM, item_vectors, item_biases, n_mid):
            query = context_his_eb
            query = self.query_layer(query)  # [1, 50, 64]
            # print(f'query_prelu: {query.shape}')
            query = self.query_prelu(query)

            inputs = paddle.concat([query, item_his_eb, query-item_his_eb, query*item_his_eb], axis=-1)  # B,T,E
            att_layer1 = self.att_layer1_layer(inputs)
            att_layer1 = F.sigmoid(att_layer1)
            att_layer2 = self.att_layer2_layer(att_layer1)
            att_layer2 = F.sigmoid(att_layer2)
            att_layer3 = self.att_layer3_layer(att_layer2)  # B,T,1
            scores = paddle.transpose(att_layer3, [0, 2, 1])  # B,1,T

            # mask
            bool_mask = paddle.equal(mask, paddle.ones_like(mask))  # B,T
            key_masks = paddle.unsqueeze(bool_mask, axis=1)  # B,1,T
            paddings = paddle.ones_like(scores) * (-2 ** 32 + 1)
            scores = paddle.where(key_masks, scores, paddings)

            # tril
            scores_tile = paddle.tile(paddle.fluid.layers.reduce_sum(scores, dim=1), [1, paddle.shape(scores)[-1]])  # B, T*T
            scores_tile = paddle.reshape(scores_tile, [-1, paddle.shape(scores)[-1], paddle.shape(scores)[-1]])  # B, T, T
            diag_vals = paddle.ones_like(scores_tile)  # B, T, T
            tril = paddle.tril(diag_vals)
            paddings = paddle.ones_like(tril) * (-2 ** 32 + 1)
            scores_tile = paddle.where(paddle.equal(tril, paddle.to_tensor(0.0)), paddings, scores_tile)  # B, T, T
            scores_tile = F.softmax(scores_tile)  # B, T, T
            att_dm_item_his_eb = paddle.matmul(scores_tile, item_his_eb)  # B, T, E

            dnn_layer1 = self.dnn_layer1_layer(att_dm_item_his_eb)
            # print(f'dnn_layer1_prelu: {dnn_layer1.shape}')
            dnn_layer1 = self.dnn_layer1_prelu(dnn_layer1)

            # target mask
            user_vector = dnn_layer1[:, -1, :]  # B, E
            user_vector2 = dnn_layer1[:, -2, :] * paddle.reshape(match_mask, [-1, paddle.shape(match_mask)[1], 1])[:, -2, :]  # B, E
            num_sampled = 2000
            labels = paddle.reshape(mid_his_batch[:, -1], [-1, 1])  # B, 1

            # not sample, slow
            # [B, E] * [E_size, cate_size]
            logits = paddle.matmul(user_vector2, item_vectors, transpose_y=True)
            logits = paddle.add(logits, item_biases)
            loss = F.softmax_with_cross_entropy(logits=logits, label=labels)

            # # sample, maybe wrong
            # logits = self.logits_layer(user_vector2)
            # logits = paddle.add(logits, item_biases)
            # loss = paddle.fluid.layers.sampled_softmax_with_cross_entropy(logits=logits, label=labels,
            #                                                               num_samples=num_sampled)

            # reduce mean batch loss
            loss = paddle.fluid.layers.reduce_mean(loss)

            return loss, user_vector, scores

        def dmr_fcn_attention(item_eb, item_his_eb, context_his_eb, mask, mode='SUM'):
            mask = paddle.equal(mask, paddle.ones_like(mask))
            item_eb_tile = paddle.tile(item_eb, [1, paddle.shape(mask)[1]]) # B, T*E
            item_eb_tile = paddle.reshape(item_eb_tile, [-1, paddle.shape(mask)[1], item_eb.shape[-1]]) # B, T, E
            if context_his_eb is None:
                query = item_eb_tile
            else:
                query = paddle.concat([item_eb_tile, context_his_eb], axis=-1)
            query = self.query_layer2(query)
            # print(f'query_prelu2: {query.shape}')
            query = self.query_prelu2(query)
            dmr_all = paddle.concat([query, item_his_eb, query-item_his_eb, query*item_his_eb], axis=-1)
            att_layer_1 = self.att_layer1_layer2(dmr_all)
            att_layer_1 = F.sigmoid(att_layer_1)
            att_layer_2 = self.att_layer2_layer2(att_layer_1)
            att_layer_2 = F.sigmoid(att_layer_2)
            att_layer_3 = self.att_layer3_layer2(att_layer_2)  # B, T, 1
            att_layer_3 = paddle.reshape(att_layer_3, [-1, 1, paddle.shape(item_his_eb)[1]])  # B,1,T
            scores = att_layer_3

            # Mask
            key_masks = paddle.unsqueeze(mask, 1)  # B,1,T
            paddings = paddle.ones_like(scores) * (-2 ** 32 + 1)
            paddings_no_softmax = paddle.zeros_like(scores)
            scores = paddle.where(key_masks, scores, paddings)  # [B, 1, T]
            scores_no_softmax = paddle.where(key_masks, scores, paddings_no_softmax)

            scores = F.softmax(scores)

            if mode == 'SUM':
                output = paddle.matmul(scores, item_his_eb)  # [B, 1, H]
                output = paddle.fluid.layers.reduce_sum(output, dim=1)  # B,E
            else:
                scores = paddle.reshape(scores, [-1, paddle.shape(item_his_eb)[1]])
                output = item_his_eb * paddle.unsqueeze(scores, -1)
                output = paddle.reshape(output, paddle.shape(item_his_eb))

            return output, scores, scores_no_softmax

        self._deep_match = deep_match
        self._dmr_fcn_attention = dmr_fcn_attention

        self.dm_item_vectors_var = paddle.nn.Embedding(
            self.cate_size,
            self.main_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="DmItemSparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.dm_item_biases = paddle.zeros(shape=[self.cate_size], dtype='float32')

        self.inp_layer = paddle.nn.BatchNorm(459, momentum=0.99, epsilon=1e-03)
        self.dnn0_layer = paddle.nn.Linear(459, 512, name='f0')
        self.dnn0_prelu = paddle.nn.PReLU(num_parameters=512, init=0.1, name='prelu0')
        self.dnn1_layer = paddle.nn.Linear(512, 256, name='f1')
        self.dnn1_prelu = paddle.nn.PReLU(num_parameters=256, init=0.1, name='prelu1')
        self.dnn2_layer = paddle.nn.Linear(256, 128, name='f2')
        self.dnn2_prelu = paddle.nn.PReLU(num_parameters=128, init=0.1, name='prelu2')
        self.dnn3_layer = paddle.nn.Linear(128, 1, name='f3')
        self.dnn3_prelu = paddle.nn.PReLU(num_parameters=1, init=0.1, name='prelu3')

    def forward(self, inputs_tensor, with_label=0):
        # input
        inputs = inputs_tensor[0]  # sparse_tensor
        dense_tensor = inputs_tensor[1]
        self.btag_his = inputs[:, 0:50]
        self.cate_his = inputs[:, 50:100]
        self.brand_his = inputs[:, 100:150]
        self.mask = inputs[:, 150:200]
        self.match_mask = inputs[:, 200:250]

        self.uid = inputs[:, 250]
        self.cms_segid = inputs[:, 251]
        self.cms_group_id = inputs[:, 252]
        self.final_gender_code = inputs[:, 253]
        self.age_level = inputs[:, 254]
        self.pvalue_level = inputs[:, 255]
        self.shopping_level = inputs[:, 256]
        self.occupation = inputs[:, 257]
        self.new_user_class_level = inputs[:, 258]

        self.mid = inputs[:, 259]
        self.cate_id = inputs[:, 260]
        self.campaign_id = inputs[:, 261]
        self.customer = inputs[:, 262]
        self.brand = inputs[:, 263]
        self.price = dense_tensor.astype('float32')

        self.pid = inputs[:, 265]

        if with_label == 1:
            self.labels = inputs[:, 266]

        # embedding layer
        self.uid_batch_embedded = self.uid_embeddings_var(self.uid)
        # self.uid_batch_embedded = paddle.reshape(self.uid_batch_embedded, shape=[-1, self.main_embedding_size])
        self.mid_batch_embedded = self.mid_embeddings_var(self.mid)
        self.cat_batch_embedded = self.cat_embeddings_var(self.cate_id)
        self.cat_his_batch_embedded = self.cat_embeddings_var(self.cate_his)
        self.brand_batch_embedded = self.brand_embeddings_var(self.brand)
        self.brand_his_batch_embedded = self.brand_embeddings_var(self.brand_his)
        self.btag_his_batch_embedded = self.btag_embeddings_var(self.btag_his)
        self.dm_btag_his_batch_embedded = self.dm_btag_embeddings_var(self.btag_his)
        self.campaign_id_batch_embedded = self.campaign_id_embeddings_var(self.campaign_id)
        self.customer_batch_embedded = self.customer_embeddings_var(self.customer)
        self.cms_segid_batch_embedded = self.cms_segid_embeddings_var(self.cms_segid)
        self.cms_group_id_batch_embedded = self.cms_group_id_embeddings_var(self.cms_group_id)
        self.final_gender_code_batch_embedded = self.final_gender_code_embeddings_var(self.final_gender_code)
        self.age_level_batch_embedded = self.age_level_embeddings_var(self.age_level)
        self.pvalue_level_batch_embedded = self.pvalue_level_embeddings_var(self.pvalue_level)
        self.shopping_level_batch_embedded = self.shopping_level_embeddings_var(self.shopping_level)
        self.occupation_batch_embedded = self.occupation_embeddings_var(self.occupation)
        self.new_user_class_level_batch_embedded = self.new_user_class_level_embeddings_var(self.new_user_class_level)
        self.pid_batch_embedded = self.pid_embeddings_var(self.pid)

        self.user_feat = paddle.concat([self.uid_batch_embedded, self.cms_segid_batch_embedded, self.cms_group_id_batch_embedded, self.final_gender_code_batch_embedded, self.age_level_batch_embedded, self.pvalue_level_batch_embedded, self.shopping_level_batch_embedded, self.occupation_batch_embedded, self.new_user_class_level_batch_embedded], -1)
        self.item_his_eb = paddle.concat([self.cat_his_batch_embedded, self.brand_his_batch_embedded], -1)
        self.item_his_eb_sum = paddle.fluid.layers.reduce_sum(self.item_his_eb, 1)
        self.item_feat = paddle.concat([self.mid_batch_embedded, self.cat_batch_embedded, self.brand_batch_embedded, self.campaign_id_batch_embedded, self.customer_batch_embedded, self.price], -1)
        self.item_eb = paddle.concat([self.cat_batch_embedded, self.brand_batch_embedded], -1)
        self.context_feat = self.pid_batch_embedded

        self.position_his_eb = self.position_embeddings_var(self.position_his)  # T, E
        self.position_his_eb = paddle.tile(self.position_his_eb, [paddle.shape(self.mid)[0], 1])  # B*T, E
        self.position_his_eb = paddle.reshape(self.position_his_eb, [paddle.shape(self.mid)[0], -1, paddle.shape(self.position_his_eb)[1]])  # B, T, E

        self.dm_position_his_eb = self.dm_position_embeddings_var(self.dm_position_his)  # T, E
        self.dm_position_his_eb = paddle.tile(self.dm_position_his_eb, [paddle.shape(self.mid)[0], 1])  # B*T, E
        self.dm_position_his_eb = paddle.reshape(self.dm_position_his_eb, [paddle.shape(self.mid)[0], -1, paddle.shape(self.dm_position_his_eb)[1]])  # B, T, E

        self.position_his_eb = paddle.concat([self.position_his_eb, self.btag_his_batch_embedded], -1)
        self.dm_position_his_eb = paddle.concat([self.dm_position_his_eb, self.dm_btag_his_batch_embedded], -1)

        # User-to-Item Network
        # Auxiliary Match Network
        self.aux_loss, self.dm_user_vector, scores = self._deep_match(self.item_his_eb, self.dm_position_his_eb, self.mask, paddle.cast(self.match_mask, 'float32'), self.cate_his, self.main_embedding_size, self.dm_item_vectors_var.weight, self.dm_item_biases, self.cate_size)
        self.aux_loss *= 0.1
        self.dm_item_vec = self.dm_item_vectors_var(self.cate_id)
        rel_u2i = paddle.fluid.layers.reduce_sum(self.dm_user_vector * self.dm_item_vec, -1, keep_dim=True)  # B,1
        self.rel_u2i = rel_u2i

        # Item-to-Item Network
        att_outputs, alphas, scores_unnorm = self._dmr_fcn_attention(self.item_eb, self.item_his_eb, self.position_his_eb, self.mask)
        rel_i2i = paddle.unsqueeze(paddle.fluid.layers.reduce_sum(scores_unnorm, [1, 2]), -1)
        self.rel_i2i = rel_i2i
        self.scores = paddle.fluid.layers.reduce_sum(alphas, 1)

        inp = paddle.concat([self.user_feat, self.item_feat, self.context_feat, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, rel_u2i, rel_i2i, att_outputs], -1)
        # print(inp.shape)  # [1, 459]

        # build fcn net
        inp = self.inp_layer(inp)
        dnn0 = self.dnn0_layer(inp)
        # print(f'dnn0_prelu: {dnn0.shape}')
        dnn0 = self.dnn0_prelu(dnn0)
        dnn1 = self.dnn1_layer(dnn0)
        # print(f'dnn1_prelu: {dnn1.shape}')
        dnn1 = self.dnn1_prelu(dnn1)
        dnn2 = self.dnn2_layer(dnn1)
        # print(f'dnn2_prelu: {dnn2.shape}')
        dnn2 = self.dnn2_prelu(dnn2)
        dnn3 = self.dnn3_layer(dnn2)
        # print(f'dnn3_prelu: {dnn3.shape}')
        dnn3 = self.dnn3_prelu(dnn3)

        # prediction
        self.y_hat = F.sigmoid(dnn3)

        if with_label == 1:
            # Cross-entropy loss and optimizer initialization
            x = paddle.fluid.layers.reduce_sum(dnn3, 1)
            ctr_loss = paddle.fluid.layers.reduce_mean(paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x, label=self.labels.astype('float32')))
            self.ctr_loss = ctr_loss
            self.loss = self.ctr_loss + self.aux_loss

            return self.y_hat, self.loss
        else:
            return self.y_hat
```


```python
%%writefile dygraph_model.py
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

import net


class DygraphModel():
    # define model
    def create_model(self, config):
        user_size = config.get("hyper_parameters.user_size")
        lr = config.get("hyper_parameters.lr")
        global_step = config.get("hyper_parameters.global_step")
        cms_segid_size = config.get("hyper_parameters.cms_segid_size")
        cms_group_id_size = config.get("hyper_parameters.cms_group_id_size")
        final_gender_code_size = config.get("hyper_parameters.final_gender_code_size")
        age_level_size = config.get("hyper_parameters.age_level_size")
        pvalue_level_size = config.get("hyper_parameters.pvalue_level_size")
        shopping_level_size = config.get("hyper_parameters.shopping_level_size")
        occupation_size = config.get("hyper_parameters.occupation_size")
        new_user_class_level_size = config.get("hyper_parameters.new_user_class_level_size")
        adgroup_id_size = config.get("hyper_parameters.adgroup_id_size")
        cate_size = config.get("hyper_parameters.cate_size")
        campaign_id_size = config.get("hyper_parameters.campaign_id_size")
        customer_size = config.get("hyper_parameters.customer_size")
        brand_size = config.get("hyper_parameters.brand_size")
        btag_size = config.get("hyper_parameters.btag_size")
        pid_size = config.get("hyper_parameters.pid_size")
        main_embedding_size = config.get("hyper_parameters.main_embedding_size")
        other_embedding_size = config.get("hyper_parameters.other_embedding_size")

        dmr_model = net.DMRLayer(user_size, lr, global_step,
                cms_segid_size, cms_group_id_size, final_gender_code_size,
                age_level_size, pvalue_level_size, shopping_level_size,
                occupation_size, new_user_class_level_size, adgroup_id_size,
                cate_size, campaign_id_size, customer_size,
                brand_size, btag_size, pid_size,
                main_embedding_size, other_embedding_size)

        return dmr_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        b = batch_data[0]
        sparse_tensor = b.astype('int64')
        for i in range(b.shape[1]):
            if i == 264:
                dense_tensor = paddle.to_tensor(b[:, i].numpy().astype(
                    'float32').reshape(-1, 1))
        label = sparse_tensor[:, -1].reshape([-1, 1])
        return label, [sparse_tensor, dense_tensor]

    # # define loss function by predicts and label
    # def create_loss(self, pred, label):
    #     cost = paddle.nn.functional.log_loss(
    #         input=pred, label=paddle.cast(
    #             label, dtype="float32"))
    #     avg_cost = paddle.mean(x=cost)
    #     return avg_cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        if config.get("hyper_parameters.optimizer.class", 'Adam') == 'Adam':
            lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
            optimizer = paddle.optimizer.Adam(
                learning_rate=lr, parameters=dy_model.parameters())
        elif config.get("hyper_parameters.optimizer.class") == 'SGD':
            lr = paddle.optimizer.lr.StepDecay(learning_rate=0.001, step_size=10000, gamma=0.8, verbose=True)
            optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        label, input_tensor = self.create_feeds(batch_data, config)

        pred, loss = dy_model(input_tensor, 1)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        # print_dict format :{'loss': loss}
        print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, input_tensor = self.create_feeds(batch_data, config)

        pred = dy_model(input_tensor, 0)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        return metrics_list, None
```


```python
%%writefile ../../../tools/utils/utils_single.py
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import envs
import os
import copy
import subprocess
import sys
import argparse
import warnings
import logging
import paddle
import numpy as np
from paddle.io import DistributedBatchSampler, DataLoader

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def _mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def get_abs_model(model):
    if model.startswith("paddlerec."):
        dir = envs.paddlerec_adapter(model)
        path = os.path.join(dir, "config.yaml")
    else:
        if not os.path.isfile(model):
            raise IOError("model config: {} invalid".format(model))
        path = model
    return path


def get_all_inters_from_yaml(file, filters):
    _envs = envs.load_yaml(file)
    all_flattens = {}

    def fatten_env_namespace(namespace_nests, local_envs):
        for k, v in local_envs.items():
            if isinstance(v, dict):
                nests = copy.deepcopy(namespace_nests)
                nests.append(k)
                fatten_env_namespace(nests, v)
            elif (k == "dataset" or k == "phase" or
                  k == "runner") and isinstance(v, list):
                for i in v:
                    if i.get("name") is None:
                        raise ValueError("name must be in dataset list. ", v)
                    nests = copy.deepcopy(namespace_nests)
                    nests.append(k)
                    nests.append(i["name"])
                    fatten_env_namespace(nests, i)
            else:
                global_k = ".".join(namespace_nests + [k])
                all_flattens[global_k] = v

    fatten_env_namespace([], _envs)
    ret = {}
    for k, v in all_flattens.items():
        for f in filters:
            if k.startswith(f):
                ret[k] = v
    return ret


def create_data_loader(config, place, mode="train"):
    if mode == "train":
        data_dir = config.get("runner.train_data_dir", None)
        batch_size = config.get('runner.train_batch_size', None)
        reader_path = config.get('runner.train_reader_path', 'reader')
    else:
        data_dir = config.get("runner.test_data_dir", None)
        batch_size = config.get('runner.infer_batch_size', None)
        reader_path = config.get('runner.infer_reader_path', 'reader')
    config_abs_dir = config.get("config_abs_dir", None)
    # data_dir = os.path.join(config_abs_dir, data_dir)
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    user_define_reader = config.get('runner.user_define_reader', False)
    logger.info("reader path:{}".format(reader_path))
    from importlib import import_module
    reader_class = import_module(reader_path)
    dataset = reader_class.RecDataset(file_list, config=config)
    loader = DataLoader(
        dataset, batch_size=batch_size, places=place, drop_last=False)
    return loader


def load_dy_model_class(abs_dir):
    sys.path.append(abs_dir)
    from dygraph_model import DygraphModel
    dy_model = DygraphModel()
    return dy_model


def load_static_model_class(config):
    abs_dir = config['config_abs_dir']
    sys.path.append(abs_dir)
    from static_model import StaticModel
    static_model = StaticModel(config)
    return static_model


def load_yaml(yaml_file, other_part=None):
    part_list = ["workspace", "runner", "hyper_parameters"]
    if other_part:
        part_list += other_part
    running_config = get_all_inters_from_yaml(yaml_file, part_list)
    return running_config


def reset_auc(auc_num=1):
    # for static clear auc
    auc_var_name = []
    for i in range(auc_num * 4):
        auc_var_name.append("_generated_var_{}".format(i))

    for name in auc_var_name:
        param = paddle.fluid.global_scope().var(name)
        if param == None:
            continue
        tensor = param.get_tensor()
        if param:
            tensor_array = np.zeros(tensor._get_dims()).astype("int64")
            tensor.set(tensor_array, paddle.CPUPlace())
            logger.info("AUC Reset To Zero: {}".format(name))
```


```python
%%writefile ../../../tools/trainer.py
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import os
import paddle.nn as nn
import time
import logging
import sys
import importlib

__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader
from utils.save_load import load_model, save_model, save_jit_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    parser.add_argument("-o", "--opt", nargs='*', type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def infer_test(dy_model, test_dataloader, dy_model_class, config, print_interval, epoch_id):
    metric_list, metric_list_name = dy_model_class.create_metrics()
    paddle.seed(12345)
    dy_model.eval()
    interval_begin = time.time()
    for batch_id, batch in enumerate(test_dataloader()):
        batch_size = len(batch[0])

        metric_list, tensor_print_dict = dy_model_class.infer_forward(
            dy_model, metric_list, batch, config)

        # if batch_id == print_interval:
        #     tensor_print_str = ""
        #     if tensor_print_dict is not None:
        #         for var_name, var in tensor_print_dict.items():
        #             tensor_print_str += (
        #                 "{}:".format(var_name) + str(var.numpy()) + ",")

        #     metric_str = ""
        #     for metric_id in range(len(metric_list_name)):
        #         metric_str += (
        #             metric_list_name[metric_id] +
        #             ": {:.6f},".format(metric_list[metric_id].accumulate())
        #         )
        #     logger.info("validation epoch: {}, batch_id: {}, ".format(
        #         epoch_id, batch_id) + metric_str + tensor_print_str +
        #                 " speed: {:.2f} ins/s".format(
        #                     print_interval * batch_size / (time.time(
        #                     ) - interval_begin)))
        #     break

    metric_str = ""
    for metric_id in range(len(metric_list_name)):
        metric_str += (
            metric_list_name[metric_id] +
            ": {:.6f},".format(metric_list[metric_id].accumulate()))

    tensor_print_str = ""
    if tensor_print_dict is not None:
        for var_name, var in tensor_print_dict.items():
            tensor_print_str += (
                "{}:".format(var_name) + str(var.numpy()) + ",")

    logger.info("validation epoch: {} done, ".format(epoch_id) + metric_str +
                tensor_print_str + " epoch time: {:.2f} s".format(
                    time.time() - interval_begin))

    dy_model.train()
    return metric_list[0].accumulate()


def _create_optimizer(dy_model, lr=0.001):
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr, parameters=dy_model.parameters())
    return optimizer


def main(args, lr):
    # paddle.seed(12345)
    # load config
    config = load_yaml(args.config_yaml)
    dy_model_class = load_dy_model_class(args.abs_dir)
    config["config_abs_dir"] = args.abs_dir
    # modify config from command
    if args.opt:
        for parameter in args.opt:
            parameter = parameter.strip()
            key, value = parameter.split("=")
            config[key] = value

    # tools.vars
    use_gpu = config.get("runner.use_gpu", True)
    use_visual = config.get("runner.use_visual", False)
    train_data_dir = config.get("runner.train_data_dir", None)
    epochs = config.get("runner.epochs", None)
    print_interval = config.get("runner.print_interval", None)
    train_batch_size = config.get("runner.train_batch_size", None)
    model_save_path = config.get("runner.model_save_path", "model_output")
    model_init_path = config.get("runner.model_init_path", None)
    save_checkpoint_interval = config.get("runner.save_checkpoint_interval", 1)

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, use_visual: {}, train_batch_size: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}, save_checkpoint_interval: {}".
            format(use_gpu, use_visual, train_batch_size, train_data_dir, epochs,
                   print_interval, model_save_path, save_checkpoint_interval))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_model_class.create_model(config)
    # print(paddle.summary(dy_model, (256, 1, 267), dtypes='int64'))

    # Create a log_visual object and store the data in the path
    if use_visual:
        from visualdl import LogWriter
        log_visual = LogWriter(args.abs_dir + "/visualDL_log/train")

    if model_init_path is not None:
        load_model(model_init_path, dy_model)

    # to do : add optimizer function
    # optimizer = dy_model_class.create_optimizer(dy_model, config)
    optimizer = _create_optimizer(dy_model, lr)

    logger.info("read data")
    train_dataloader = create_data_loader(config=config, place=place)
    test_dataloader = create_data_loader(config=config, place=place, mode="test")

    last_epoch_id = config.get("last_epoch", -1)
    step_num = 0

    best_metric = 0

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        dy_model.train()
        metric_list, metric_list_name = dy_model_class.create_metrics()
        # auc_metric = paddle.metric.Auc("ROC")
        epoch_begin = time.time()
        interval_begin = time.time()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()

        for batch_id, batch in enumerate(train_dataloader()):
            train_reader_cost += time.time() - reader_start
            optimizer.clear_grad()
            train_start = time.time()
            batch_size = len(batch[0])

            loss, metric_list, tensor_print_dict = dy_model_class.train_forward(
                dy_model, metric_list, batch, config)

            # print(loss)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (
                            metric_list_name[metric_id] +
                            ":{:.6f}, ".format(metric_list[metric_id].accumulate())
                    )
                    if use_visual:
                        log_visual.add_scalar(
                            tag="train/" + metric_list_name[metric_id],
                            step=step_num,
                            value=metric_list[metric_id].accumulate())
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += (
                                "{}:".format(var_name) + str(var.numpy()) + ",")
                        if use_visual:
                            log_visual.add_scalar(
                                tag="train/" + var_name,
                                step=step_num,
                                value=var.numpy())
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(
                        epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} ins/s, loss: {:.6f}".
                    format(train_reader_cost / print_interval, (
                            train_reader_cost + train_run_cost) / print_interval,
                           total_samples / print_interval, total_samples / (
                                   train_reader_cost + train_run_cost), loss.numpy()[0]))

                # if batch_id > 80000:
                #     tmp_auc = infer_test(dy_model, test_dataloader, dy_model_class, config, print_interval, epoch_id)
                #     if tmp_auc > best_metric:
                #         best_metric = tmp_auc
                #         save_model(dy_model, optimizer, model_save_path, 1000+epoch_id, prefix='rec')
                #         logger.info(f"saved best model, {metric_list_name[0]}: {best_metric}")

                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()
            step_num = step_num + 1

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (
                    metric_list_name[metric_id] +
                    ": {:.6f},".format(metric_list[metric_id].accumulate()))

        tensor_print_str = ""
        if tensor_print_dict is not None:
            for var_name, var in tensor_print_dict.items():
                tensor_print_str += (
                        "{}:".format(var_name) + str(var.numpy()) + ",")

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    tensor_print_str + " epoch time: {:.2f} s".format(
            time.time() - epoch_begin))

        # if metric_list[0].accumulate() > best_metric:
        #     best_metric = metric_list[0].accumulate()
        #     save_model(
        #         dy_model, optimizer, model_save_path, 1000, prefix='rec')  # best model
        #     # save_jit_model(dy_model, model_save_path, prefix='tostatic')
        #     logger.info(f"saved best model, {metric_list_name[0]}: {best_metric}")

        if epoch_id % save_checkpoint_interval == 0 and metric_list[0].accumulate() > 0.5:
            save_model(dy_model, optimizer, model_save_path, epoch_id, prefix='rec')  # middle epochs

        if metric_list[0].accumulate() >= 0.95:
            print('Already over fitting, stop training!')
            break

    infer_auc = infer_test(dy_model, test_dataloader, dy_model_class, config, print_interval, epoch_id)
    return infer_auc


if __name__ == '__main__':
    import os
    import shutil

    def f(best_auc, best_lr, current_lr, args):
        auc = main(args, current_lr)
        print(f'Trying Current_lr: {current_lr}, AUC: {auc}')
        if auc > best_auc:
            best_auc = auc
            best_lr = current_lr
            shutil.rmtree('output_model_dmr_fulldata/1000', ignore_errors=True)
            shutil.copytree('output_model_dmr_fulldata/0', 'output_model_dmr_fulldata/1000')
            os.rename(src='output_model_dmr_fulldata/0', dst=f'output_model_dmr_fulldata/b5120l{str(lr)[2:]}auc{str(auc)[2:]}')
            print(f'rename 0 to b5120l{str(lr)[2:]}auc{str(auc)[2:]}')
        return best_auc, best_lr

    def reset_graph():
        paddle.fluid.dygraph.disable_dygraph()
        paddle.fluid.dygraph.enable_dygraph()

    args = parse_args()
    best_auc = 0.6
    best_lr = -1
    try_lrs = [0.006, 0.007, 0.008, 0.009, 0.01] * 2

    for lr in try_lrs:
        best_auc, best_lr = f(best_auc, best_lr, lr, args)
        reset_graph()
        if best_auc >= 0.6447:
            break

    print(f'Best AUC: {best_auc}, Best learning_rate: {best_lr}')
```


```python
%%writefile ../../../tools/infer.py
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import os
import paddle.nn as nn
import time
import logging
import sys
import importlib

__dir__ = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader
from utils.save_load import save_model, load_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    parser.add_argument("-o", "--opt", nargs='*', type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(12345)
    # load config
    config = load_yaml(args.config_yaml)
    dy_model_class = load_dy_model_class(args.abs_dir)
    config["config_abs_dir"] = args.abs_dir
    # modify config from command
    if args.opt:
        for parameter in args.opt:
            parameter = parameter.strip()
            key, value = parameter.split("=")
            config[key] = value

    # tools.vars
    use_gpu = config.get("runner.use_gpu", True)
    use_visual = config.get("runner.use_visual", False)
    test_data_dir = config.get("runner.test_data_dir", None)
    print_interval = config.get("runner.print_interval", None)
    infer_batch_size = config.get("runner.infer_batch_size", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, use_visual: {}, infer_batch_size: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, use_visual, infer_batch_size, test_data_dir,
               start_epoch, end_epoch, print_interval, model_load_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_model_class.create_model(config)

    # Create a log_visual object and store the data in the path
    if use_visual:
        from visualdl import LogWriter
        log_visual = LogWriter(args.abs_dir + "/visualDL_log/infer")

    # to do : add optimizer function
    #optimizer = dy_model_class.create_optimizer(dy_model, config)

    logger.info("read data")
    test_dataloader = create_data_loader(
        config=config, place=place, mode="test")

    epoch_begin = time.time()
    interval_begin = time.time()

    metric_list, metric_list_name = dy_model_class.create_metrics()
    step_num = 0

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        try:
            load_model(model_path, dy_model)
        except Exception as e:
            print(e)
            continue
        dy_model.eval()
        infer_reader_cost = 0.0
        infer_run_cost = 0.0
        reader_start = time.time()

        for batch_id, batch in enumerate(test_dataloader()):
            infer_reader_cost += time.time() - reader_start
            infer_start = time.time()
            batch_size = len(batch[0])

            metric_list, tensor_print_dict = dy_model_class.infer_forward(
                dy_model, metric_list, batch, config)

            infer_run_cost += time.time() - infer_start

            if batch_id % print_interval == 0:
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += (
                            "{}:".format(var_name) + str(var.numpy()) + ",")
                        if use_visual:
                            log_visual.add_scalar(
                                tag="infer/" + var_name,
                                step=step_num,
                                value=var.numpy())
                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (
                        metric_list_name[metric_id] +
                        ": {:.6f},".format(metric_list[metric_id].accumulate())
                    )
                    if use_visual:
                        log_visual.add_scalar(
                            tag="infer/" + metric_list_name[metric_id],
                            step=step_num,
                            value=metric_list[metric_id].accumulate())
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(
                        epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.2f} ins/s".
                    format(infer_reader_cost / print_interval, (
                        infer_reader_cost + infer_run_cost) / print_interval,
                           infer_batch_size, print_interval * batch_size / (
                               time.time() - interval_begin)))
                interval_begin = time.time()
                infer_reader_cost = 0.0
                infer_run_cost = 0.0
            step_num = step_num + 1
            reader_start = time.time()

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (
                metric_list_name[metric_id] +
                ": {:.6f},".format(metric_list[metric_id].accumulate()))

        tensor_print_str = ""
        if tensor_print_dict is not None:
            for var_name, var in tensor_print_dict.items():
                tensor_print_str += (
                    "{}:".format(var_name) + str(var.numpy()) + ",")

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    tensor_print_str + " epoch time: {:.2f} s".format(
                        time.time() - epoch_begin))
        epoch_begin = time.time()


if __name__ == '__main__':
    args = parse_args()
    main(args)
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
current_path = os.path.abspath(os.curdir)
print(current_path)
```


```python
%%writefile config.yaml
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# global settings

runner:
  train_data_dir: "/home/aistudio/data/sample_data/train"
  train_reader_path: "alimama_reader" # importlib format
  use_gpu: True
  use_auc: True
  train_batch_size: 256
  epochs: 1
  print_interval: 50
  # model_init_path: "output_model_dmr/0" # init model
  model_save_path: "output_model_dmr"
  test_data_dir: "/home/aistudio/data/sample_data/test"
  infer_reader_path: "alimama_reader" # importlib format
  infer_batch_size: 256
  infer_load_path: "output_model_dmr"
  infer_start_epoch: 0
  infer_end_epoch: 1
  with_label: 1
  use_visual: True
  save_checkpoint_interval: 1

# hyper parameters of user-defined network
hyper_parameters:
  # optimizer config
  optimizer:
    class: Adam
    learning_rate: 0.001
    strategy: async
  # user-defined <key, value> pairs
  # user feature size
  user_size: 1141730
  cms_segid_size: 97
  cms_group_id_size: 13
  final_gender_code_size: 3
  age_level_size: 7
  pvalue_level_size: 4
  shopping_level_size: 4
  occupation_size: 3
  new_user_class_level_size: 5

  # item feature size
  adgroup_id_size: 846812
  cate_size: 12978
  campaign_id_size: 423437
  customer_size: 255876
  brand_size: 461529

  # context feature size
  btag_size: 5
  pid_size: 2

  # embedding size
  main_embedding_size: 32
  other_embedding_size: 8
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
%%writefile config_bigdata.yaml
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# global settings

runner:
  train_data_dir: "/home/aistudio/data/full_data/train/"
  train_reader_path: "alimama_reader" # importlib format
  use_gpu: True
  use_auc: True
  train_batch_size: 5120
  epochs: 1
  print_interval: 1000
  #model_init_path: "output_model_dmr_fulldata/0" # init model
  model_save_path: "output_model_dmr_fulldata"
  test_data_dir: "/home/aistudio/data/full_data/test/"
  infer_reader_path: "alimama_reader" # importlib format
  infer_batch_size: 512
  infer_load_path: "output_model_dmr_fulldata"
  infer_start_epoch: 1000
  infer_end_epoch: 1001
  with_label: 1
  use_visual: True
  save_checkpoint_interval: 1

# hyper parameters of user-defined network
hyper_parameters:
  # optimizer config
  optimizer:
    class: Adam
    learning_rate: 0.008
  # user-defined <key, value> pairs
  # user feature size
  user_size: 1141730
  cms_segid_size: 97
  cms_group_id_size: 13
  final_gender_code_size: 3
  age_level_size: 7
  pvalue_level_size: 4
  shopping_level_size: 4
  occupation_size: 3
  new_user_class_level_size: 5

  # item feature size
  adgroup_id_size: 846812
  cate_size: 12978
  campaign_id_size: 423437
  customer_size: 255876
  brand_size: 461529

  # context feature size
  btag_size: 5
  pid_size: 2

  # embedding size
  main_embedding_size: 32
  other_embedding_size: 8
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
