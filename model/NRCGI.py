#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as metrics
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from layers import Dice
from layers import MultiLayerPerceptron
import torch.nn.functional as F


class CrossAttention(nn.Module):
    
    def __init__(self, embed_dim=8, hidden_dim=8):
        super(CrossAttention, self).__init__()
        self.w_left = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_right = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_tot = nn.Linear(2*hidden_dim, 1)
        self.get_score = nn.Tanh()
        
    def forward(self, source, target):
        trans_left = self.w_left(torch.abs(torch.sub(source, target))) # 1024*clusters*10
        trans_right = self.w_right(torch.mul(source, target)) #1024*clusters*10
        att = self.w_tot(torch.cat([trans_left, trans_right], dim=2))
        score = self.get_score(att)
        return score

    
class NRCGI(nn.Module):
    
    def __init__(self, field_dims, cate_list, u_cluster_list, i_cluster_list, cluster_num, seq_len, sample_size, embed_dim=8, batch_size=256, device='cpu'):
        super(NRCGI, self).__init__()
        # user和item embedding层
        self.user_embed = nn.Embedding(field_dims[0], embed_dim)
        self.item_embed = nn.Embedding(field_dims[1], embed_dim)
        self.cate_embed = nn.Embedding(field_dims[2], embed_dim)
        torch.nn.init.xavier_uniform_(self.user_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.item_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.cate_embed.weight.data)
        self.u_l1_attention = CrossAttention(embed_dim, 2*embed_dim)
        self.i_l1_attention = CrossAttention(embed_dim, 2*embed_dim)
        self.u_l2_attention = CrossAttention(embed_dim, 2*embed_dim)
        self.i_l2_attention = CrossAttention(embed_dim, 2*embed_dim)
        self.cluster_num = cluster_num
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.cate_list = cate_list
        self.u_cluster_list = u_cluster_list
        self.i_cluster_list = i_cluster_list
        self.seq_len = seq_len
        self.sample_size = sample_size
        self.cluster_map1 = torch.arange(1, self.cluster_num+1, requires_grad=False).unsqueeze_(-1).expand(self.cluster_num, self.seq_len).to(device) # [cluster_num, 1]
        self.cluster_map2 = torch.arange(1, self.cluster_num+1, requires_grad=False).unsqueeze_(-1).expand(self.cluster_num, self.seq_len*self.sample_size).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 15, 200),
            Dice(),
            nn.Linear(200, 80),
            Dice(),
            nn.Linear(80, 1)
        )
        
    def forward(self, x):
        # 0, 1-100, 101-400, 401, 402-501,502-801
        user_embedding = self.user_embed(x[:,0]) # 1024 * 8
        #user_time_embedding = self.time_embed()
        user_behaviors_ad = x[:, 1: self.seq_len+1] # 1024 * 100
        user_behaviors_cluster = self.i_cluster_list[user_behaviors_ad] # 1024 * 100
        user_behaviors_cate = self.cate_list[user_behaviors_ad]
        
        user_layer3_user = x[:, self.seq_len+1:self.seq_len+self.seq_len*self.sample_size+1]
        user_layer3_cluster = self.u_cluster_list[user_layer3_user]
        
        item_embedding = torch.sum(torch.stack([self.item_embed(x[:,self.seq_len+self.seq_len*self.sample_size+1]), self.cate_embed(self.cate_list[x[:,self.seq_len+self.seq_len*self.sample_size+1]])], 0), dim=0) # 1024 * 8
        item_history_us = x[:,self.seq_len+self.seq_len*self.sample_size+2:2*self.seq_len+self.seq_len*self.sample_size+2] # 1024 * 100
        item_history_cluster = self.u_cluster_list[item_history_us] # 1024 * 100
        
        item_layer3_item = x[:, 2*self.seq_len+self.seq_len*self.sample_size+2:]
        item_layer3_cluster = self.i_cluster_list[item_layer3_item]
        item_layer3_cate = self.cate_list[item_layer3_item]
        
        u_l1_emb_init = torch.sum(torch.stack([self.item_embed(user_behaviors_ad), 
                                         self.cate_embed(user_behaviors_cate)], 0), dim=0)
        
        u_l2_emb_init = self.user_embed(user_layer3_user)
        
        i_l1_emb_init = self.user_embed(item_history_us)
        
        i_l2_emb_init = torch.sum(torch.stack([self.item_embed(item_layer3_item), 
                                         self.cate_embed(item_layer3_cate)], 0), dim=0)
        
        ############## 建立cluster mask ###################
        cluster_map_l1 = self.cluster_map1.expand(x.shape[0], self.cluster_num, self.seq_len) # [1024, 5, 100]
        cluster_map_l2 = self.cluster_map2.expand(x.shape[0], self.cluster_num, self.seq_len*self.sample_size) # [1024, 5, 200]
        #print(self.cluster_map.shape, cluster_map_l1.shape)
        #print(cluster_map_l1)

        u_l1_batch_mask = (user_behaviors_cluster.unsqueeze(1) == cluster_map_l1)
        i_l1_batch_mask = (item_history_cluster.unsqueeze(1) == cluster_map_l1)
        u_l2_batch_mask = (user_layer3_cluster.unsqueeze(1) == cluster_map_l2)
        i_l2_batch_mask = (item_layer3_cluster.unsqueeze(1) == cluster_map_l2)

        u_l1_emb_exp = u_l1_emb_init.unsqueeze(1).repeat(1, self.cluster_num, 1, 1)
        i_l1_emb_exp = i_l1_emb_init.unsqueeze(1).repeat(1, self.cluster_num, 1, 1)
        u_l2_emb_exp = u_l2_emb_init.unsqueeze(1).repeat(1, self.cluster_num, 1, 1)
        i_l2_emb_exp = i_l2_emb_init.unsqueeze(1).repeat(1, self.cluster_num, 1, 1)
        
        #print(u_l1_batch_mask, u_l1_batch_mask.shape)
        
        u_l1_embs = u_l1_emb_exp.mul(u_l1_batch_mask.unsqueeze(-1)).sum(dim=2)
        u_l2_embs = u_l2_emb_exp.mul(u_l2_batch_mask.unsqueeze(-1)).sum(dim=2)
        i_l1_embs = i_l1_emb_exp.mul(i_l1_batch_mask.unsqueeze(-1)).sum(dim=2)
        i_l2_embs = i_l2_emb_exp.mul(i_l2_batch_mask.unsqueeze(-1)).sum(dim=2)
        
        #print(u_l1_embs, u_l1_embs.shape)
        
        u_l1_att = self.u_l1_attention(u_l1_embs, item_embedding.unsqueeze(1))  # (batch_size, num_behaviors, 1)  
        u_l2_att = self.u_l2_attention(u_l2_embs, user_embedding.unsqueeze(1))  # (batch_size, num_behaviors, 1)  
        
        i_l1_att = self.i_l1_attention(i_l1_embs, user_embedding.unsqueeze(1))
        i_l2_att = self.i_l2_attention(i_l2_embs, item_embedding.unsqueeze(1))
        
        ######################### 得到兴趣embedding #########################
        u_l1_rep = u_l1_embs.mul(u_l1_att).sum(dim=1)
        u_l2_rep = u_l2_embs.mul(u_l2_att).sum(dim=1)
        
        i_l1_rep = i_l1_embs.mul(i_l1_att).sum(dim=1)
        i_l2_rep = i_l2_embs.mul(i_l2_att).sum(dim=1)
        
        cross_emb1 = user_embedding.mul(item_embedding)
        cross_emb2 = user_embedding.mul(i_l1_rep)
        cross_emb3 = user_embedding.mul(i_l2_rep)
        
        cross_emb4 = u_l1_rep.mul(item_embedding)
        cross_emb5 = u_l1_rep.mul(i_l1_rep)
        cross_emb6 = u_l1_rep.mul(i_l2_rep)
        
        cross_emb7 = u_l2_rep.mul(item_embedding)
        cross_emb8 = u_l2_rep.mul(i_l1_rep)
        cross_emb9 = u_l2_rep.mul(i_l2_rep)
        
        ####################### Feature crossing ########################
        concated = torch.hstack([user_embedding, item_embedding, u_l1_rep, i_l1_rep, u_l2_rep, i_l2_rep, cross_emb1, cross_emb2, cross_emb3, cross_emb4, cross_emb5,
                                cross_emb6, cross_emb7, cross_emb8, cross_emb9])
        output = self.mlp(concated)
        output = torch.sigmoid(output)
        return output
