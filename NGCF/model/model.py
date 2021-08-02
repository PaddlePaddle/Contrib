# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
import paddle.nn as nn
from dataloader import BasicDataset
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import config
import pgl
from pgl.nn import functional as GF
from tqdm import tqdm

class BasicModel(nn.Layer):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
def degree_norm(graph, mode="indegree", p=-1):
    """Calculate the degree normalization of a graph
    Args:
        graph: the graph object from (:code:`Graph`)
        mode: which degree to be normalized ("indegree" or "outdegree")
    return:
        A tensor with shape (num_nodes, 1).
    """

    assert mode in [
        'indegree', 'outdegree'
    ], "The degree_norm mode should be in ['indegree', 'outdegree']. But recieve mode=%s" % mode

    if mode == "indegree":
        degree = graph.indegree()+1
    elif mode == "outdegree":
        degree = graph.outdegree()+1

    norm = paddle.cast(degree, dtype=paddle.get_default_dtype())
    norm = paddle.clip(norm, min=1.0)
    norm = paddle.pow(norm, p)
    norm = paddle.reshape(norm, [-1, 1])
    return norm
class CustomGCNConv(nn.Layer):
    def __init__(self, input_size, output_size):
        super(CustomGCNConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform(fan_in=1, fan_out=output_size))
        self.linear = nn.Linear(input_size, output_size, weight_attr, bias_attr)
    def forward(self, graph, feature):

        norm = degree_norm(graph)
        ouput = graph.send_recv(feature, "sum")
        ouput = ouput + feature
        ouput = ouput * norm
        output1 = self.linear(ouput)
        return output1, ouput
    
class NGCFConv(nn.Layer):
    def __init__(self, input_size, output_size, n_layers=3):
        super(NGCFConv, self).__init__()
        self.n_layers = n_layers
        self.gcn = nn.LayerList()
        self.linear = nn.LayerList()
        self.mess_dropout = []
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform(fan_in=1, fan_out=output_size))
        for i in range(n_layers):
            self.gcn.append(CustomGCNConv(input_size, output_size))
            self.linear.append(nn.Linear(input_size, output_size, weight_attr, bias_attr))
            self.mess_dropout.append(0.1)
    def forward(self, graph:pgl.Graph, user_feature:nn.Embedding, item_feature:nn.Embedding):
        ego_embeddings = paddle.concat([user_feature, item_feature])
        
        embs = [ego_embeddings]
        for i in range(self.n_layers):
            sum_embeddings, side_embeddings = self.gcn[i](graph, ego_embeddings)
            bi_embeddings = paddle.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = self.linear[i](bi_embeddings)
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            # ego_embeddings = nn.Dropout(self.mess_dropout[i])(ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, axis=1)
            embs.append(norm_embeddings)
        embs = paddle.concat(embs, axis=1)
        users, items = paddle.split(embs, [user_feature.shape[0], item_feature.shape[0]])
        return users, items    
class NGCF(nn.Layer):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(NGCF, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.gcn = NGCFConv(self.latent_dim, self.latent_dim, self.n_layers)
        weight_attr1 = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        weight_attr2 = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        self.embedding_user = nn.Embedding(
                num_embeddings=self.num_users, embedding_dim=self.latent_dim, weight_attr=weight_attr1)
        self.embedding_item = nn.Embedding(
                num_embeddings=self.num_items, embedding_dim=self.latent_dim, weight_attr=weight_attr2)
        self.f = nn.Sigmoid()
        num_nodes = self.dataset.n_users + self.dataset.m_items
        edges = paddle.to_tensor(self.dataset.trainEdge, dtype='int64')

        self.Graph = pgl.Graph(num_nodes=num_nodes, edges=edges)
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
    def getUsersRating(self, users):

        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_users, all_items = self.gcn(self.Graph, users_emb, items_emb)

        users_emb = paddle.to_tensor(all_users.numpy()[users.numpy()])
        items_emb = all_items
        rating = self.f(paddle.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_users, all_items = self.gcn(self.Graph, users_emb, items_emb)
        users_emb = paddle.index_select(all_users, users)
        pos_emb = paddle.index_select(all_items, pos_items)
        neg_emb = paddle.index_select(all_items, neg_items)
        return users_emb, pos_emb, neg_emb
    
    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb = self.getEmbedding(users.astype('int32'), pos.astype('int32'), neg.astype('int32'))
        
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                         pos_emb.norm(2).pow(2)  +
                         neg_emb.norm(2).pow(2))/float(len(users))
        pos_scores = paddle.multiply(users_emb, pos_emb)
        pos_scores = paddle.sum(pos_scores, axis=1)
        neg_scores = paddle.multiply(users_emb, neg_emb)
        neg_scores = paddle.sum(neg_scores, axis=1)
        
        loss = nn.LogSigmoid()(pos_scores - neg_scores)

        loss = -1 * paddle.mean(loss)
        return loss, reg_loss
    
    def forward(self, users, items):

        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_users, all_items = self.lgn(self.Graph, users_emb, items_emb)
        users_emb = paddle.to_tensor(all_users.numpy()[users.numpy()])
        items_emb = paddle.to_tensor(all_items.numpy()[items.numpy()])
        inner_pro = paddle.multiply(users_emb, items_emb)
        gamma     = paddle.sum(inner_pro, axis=1)
        return gamma