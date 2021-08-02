#encoding=utf8
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
class LightGCNonv(nn.Layer):
    def __init__(self, n_layers):
        super(LightGCNonv, self).__init__()
        self.n_layers = n_layers
    def forward(self, graph, user_feature, item_feature, norm=None):
        """
        propagate methods for lightGCN
        """
        
        norm = GF.degree_norm(graph)
        feature = paddle.concat([user_feature, item_feature])
        embs = [feature]
        
        for layer in range(self.n_layers):
            feature = feature * norm
            feature = graph.send_recv(feature, "sum")
            feature = feature * norm
            embs.append(feature)
        embs = paddle.stack(embs, axis=1)
        light_out = paddle.mean(embs, axis=1)
        users, items = paddle.split(light_out, [user_feature.shape[0], item_feature.shape[0]])
        return users, items
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.lgn = LightGCNonv(self.n_layers)
        self.embedding_user = nn.Embedding(
                num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(
                num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            emb_item_weight = np.random.normal(0, 0.1, self.embedding_item.weight.numpy().shape).astype(np.float32)
            emb_user_weight = np.random.normal(0, 0.1, self.embedding_user.weight.numpy().shape).astype(np.float32)
        else:
            emb_item_weight = np.load('item_embedding.npy').astype(np.float32)
            emb_user_weight = np.load('item_embedding.npy').astype(np.float32)
        self.embedding_item.weight.set_value(emb_item_weight)
        self.embedding_user.weight.set_value(emb_user_weight)

        self.f = nn.Sigmoid()
        num_nodes = self.dataset.n_users + self.dataset.m_items
        edges = paddle.to_tensor(self.dataset.trainEdge, dtype='int64')

        self.Graph = pgl.Graph(num_nodes=num_nodes, edges=edges)
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
        self.lgn.train()
    def getUsersRating(self, users):

        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_users, all_items = self.lgn(self.Graph, users_emb, items_emb)

        users_emb = paddle.to_tensor(all_users.numpy()[users.numpy()])
        items_emb = all_items
        rating = self.f(paddle.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_users, all_items = self.lgn(self.Graph, users_emb, items_emb)
        users_emb = paddle.index_select(all_users, users)
        pos_emb = paddle.index_select(all_items, pos_items)
        neg_emb = paddle.index_select(all_items, neg_items)
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.astype('int32'), pos.astype('int32'), neg.astype('int32'))
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = paddle.multiply(users_emb, pos_emb)
        pos_scores = paddle.sum(pos_scores, axis=1)
        neg_scores = paddle.multiply(users_emb, neg_emb)
        neg_scores = paddle.sum(neg_scores, axis=1)
        loss = paddle.mean(paddle.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss
    
    def forward(self, users, items):

        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_users, all_items = self.lgn(self.Graph, users_emb, items_emb)
        users_emb = paddle.index_select(all_users, users)
        items_emb = paddle.index_select(all_items, items)
        inner_pro = paddle.multiply(users_emb, items_emb)
        gamma     = paddle.sum(inner_pro, axis=1)
        return gamma