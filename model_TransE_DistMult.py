import numpy as np
import torch
from utils import *

# 距离选择l1或者l2范数

'''
这部分是Distmult和TransE的模型
'''

class transE(torch.nn.Module):
    def __init__(self, d, dim, p_norm):
        super(transE, self).__init__()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, max_norm=1, padding_idx=0).cuda()
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0).cuda()
        torch.nn.init.xavier_uniform_(self.Eh.weight.data)
        torch.nn.init.xavier_uniform_(self.rvh.weight.data)
        self.margin = 1
        self.p_norm = p_norm # 2范数或者1范数

        self.criterion = torch.nn.MarginRankingLoss(self.margin, reduction='sum')

    def forward(self, u_idx, r_idx, v_idx):
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        rvh = self.rvh.weight[r_idx]

        return torch.norm(u + rvh - v, self.p_norm, -1)


class distmult(torch.nn.Module):
    def __init__(self, d, dim):
        super(distmult, self).__init__()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, max_norm=1).cuda()
        self.rvh = torch.nn.Embedding(len(d.entities), dim).cuda() # 关系不优化
        torch.nn.init.xavier_uniform_(self.Eh.weight.data)
        torch.nn.init.xavier_uniform_(self.rvh.weight.data)
        self.margin = 2

        # self.criterion = torch.nn.Softplus()
        self.criterion = torch.nn.MarginRankingLoss(self.margin, reduction='sum')

    def forward(self, u_idx, r_idx, v_idx):
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        rvh = self.rvh.weight[r_idx]

        return -torch.sum((u * v) * rvh, -1).flatten()