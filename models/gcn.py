import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from .inits import glorot


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCNConv(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, data):
        x = data.features
        adj = data.adj
        x = torch.matmul(x, self.weight)
        x = torch.spmm(adj, x)
        if self.bias is not None:
            return x + self.bias
        else:
            return x


def create_gcn_model(data, nhid=16, dropout=0.5, lr=0.01, weight_decay=5e-4):
    model = GCN(data.num_features, nhid, data.num_classes, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
