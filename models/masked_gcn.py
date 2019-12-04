import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn.modules.module import Module

from utils import get_degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_mask(x, edge_list, dense_adj, sigma, deg):
    source, target = edge_list
    h = torch.pow((x[source] - x[target]), 2)
    h = h * dense_adj[source, target].view(-1, 1)
    mask = torch.zeros(x.size(), device=device)
    mask.index_add_(0, source, h)
    mask = mask / (sigma * sigma)
    mask = torch.exp(- mask / deg.view(-1, 1))
    return mask


class MaskedGCN(nn.Module):
    def __init__(self, data, nhid, dropout):
        super(MaskedGCN, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc1 = MaskedGCNConv(nfeat, nhid, data)
        self.gc2 = MaskedGCNConv(nhid, nclass, data)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x = data.features
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x, data))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, data)
        return F.log_softmax(x, dim=1)


class MaskedGCNConv(Module):
    def __init__(self, in_features, out_features, data, bias=True):
        super(MaskedGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.degree = get_degree(data.edge_list).float()
        self.dense_adj = data.adj.to_dense().to(device)
        self.sigma = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)
        self.sigma.data.fill_(1)

    def forward(self, x, data):
        mask = make_mask(x, data.edge_list, self.dense_adj, self.sigma, self.degree)
        x = mask * x
        x = self.fc(x)
        x = torch.spmm(data.adj, x)
        return x


def create_masked_gcn_model(data, nhid=16, dropout=0.5, lr=0.01, weight_decay=5e-4):
    model = MaskedGCN(data, nhid, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
