# -*- coding: utf-8 -*-
# @Time : 2021/12/23 13:43
# @Author : Yumo
# @File : Model.py
# @Project: GOODKT
# @Comment :

from KnowledgeTracing.hgnn_models import HGNN
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
import torch
import torch.nn.functional as F
from KnowledgeTracing.DirectedGCN.GCN import GCN
from KnowledgeTracing.BipartiteGCN.BPR import BPR

class DKT(nn.Module):

    def __init__(self, hidden_dim, layer_dim, G, adj_in, adj_out, bipartite_G):
        super(DKT, self).__init__()
        '''initial feature'''
        emb_dim = C.EMB
        emb_ques = nn.Embedding(C.item_num, emb_dim)
        emb_stus = nn.Embedding(C.user_num, emb_dim)
        # nn.init.normal_(emb_ques.weight, std=0.01)
        # nn.init.normal_(emb_stus.weight, std=0.01)
        # self.ques = emb_ques(torch.LongTensor([i for i in range(C.item_num)])).cuda()
        # self.stus = emb_stus(torch.LongTensor([i for i in range(C.user_num)])).cuda()
        self.ques = emb_ques.weight
        self.stus = emb_stus.weight

        self.emb_k = nn.Linear(emb_dim, emb_dim)
        self.emb_q = nn.Linear(emb_dim, emb_dim)
        self.emb_v = nn.Linear(emb_dim, emb_dim)

        '''generate two graphs'''
        self.G = G
        self.adj_out = adj_out
        self.adj_in = adj_in

        self.user_item_matrix = bipartite_G[0]
        self.item_user_matrix = bipartite_G[1]
        self.user_item_matrix_ = bipartite_G[2]
        self.item_user_matrix_ = bipartite_G[3]

        '''DGCN'''
        self.net1 = GCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2))
        self.net2 = GCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2))

        '''HGCN'''
        self.net = HGNN(in_ch=C.EMB,
                        n_hid=C.EMB,
                        n_class=C.EMB)

        '''LightGCN'''
        self.net3 = BPR(user_num=C.user_num,
                        item_num=C.item_num,
                        factor_num=C.EMB,
                        user_item_matrix=self.user_item_matrix,
                        item_user_matrix=self.item_user_matrix)

        self.net4 = BPR(user_num=C.user_num,
                        item_num=C.item_num,
                        factor_num=C.EMB,
                        user_item_matrix=self.user_item_matrix_,
                        item_user_matrix=self.item_user_matrix_)

        '''GRU'''
        self.rnn1 = nn.GRU(C.EMB, hidden_dim, layer_dim, batch_first=True)
        self.rnn2 = nn.GRU(C.EMB, hidden_dim, layer_dim, batch_first=True)
        self.rnn3 = nn.GRU(C.EMB, hidden_dim, layer_dim, batch_first=True)
        '''kd'''
        self.fc_h = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.fc_d = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.fc_light = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.fc_ensemble = nn.Linear(2 * hidden_dim, C.NUM_OF_QUESTIONS)
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)

        self.sigmoid = nn.Sigmoid()

    def attantion(self, x):

        # x = [128, 50, 256]
        # emb_q = [128, 50, 256]
        # emb_k = [128, 50, 256]
        # emb_v = [128, 50, 256]
        # a = [128, 50, 50]
        # x = [128, 50, 256]

        q = self.emb_q(x)
        k = self.emb_k(x)
        v = self.emb_v(x)

        a = torch.matmul(q, k.transpose(1, 2))
        a = F.softmax(a,dim=2)
        x = torch.matmul(a, v)

        return x

    def forward(self, x):
        '''SkillGraph: HGCN'''
        ques_h = self.net(self.ques, self.G)
        '''TransitionGraph: DGCN'''
        # ques_out = self.net1(self.ques, self.adj_out)
        # ques_in = self.net2(self.ques, self.adj_in)
        # ques_d = torch.cat([ques_in, ques_out], -1)
        '''AbilityGraph: LightGCN'''
        ques_light = self.net3(self.stus, self.ques)
        ques_light = self.net4(self.stus, self.ques)

        '''choose 50'''
        x_h = x.matmul(ques_h)
        # x_d = x.matmul(ques_d)
        x_light = x.matmul(ques_light)

        x_h = self.attantion(x_h)
        x_light = self.attantion(x_light)

        '''gru'''
        out_h, _ = self.rnn1(x_h)
        # out_d, _ = self.rnn2(x_d)
        out_light, _ = self.rnn3(x_light)

        '''logits'''
        logit_h = self.fc_h(out_h)
        # logit_d = self.fc_d(out_d)
        logit_light = self.fc_light(out_light)

        '''kd'''
        theta = self.sigmoid(self.w1(out_h) + self.w2(out_light))
        out_d = theta * out_light
        out_h = (1 - theta) * out_h
        emseble_logit = self.fc_ensemble(torch.cat([out_h, out_light], -1))

        # emseble_logit = self.fc_ensemble((1/2)*torch.cat([out_h, out_light], -1))

        return logit_h, logit_light, emseble_logit