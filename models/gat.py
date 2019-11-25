import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from .inits import glorot


class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.gc1 = GATConv()
        self.gc2 = GATConv()


class GATConv(Module):
    def __init__(self, in_features, out_features, dropout, alpha, bias=True, concat=True):
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
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

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(x)
        return F.log_softmax(x, dim=1)

