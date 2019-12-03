import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.modules.module import Module


class GCN(nn.Module):
    def __init__(self, data, nhid, dropout):
        super(GCN, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x = data.features
        adj = data.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCNConv(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def forward(self, x, adj):
        x = self.fc(x)
        x = torch.spmm(adj, x)
        return x


def create_gcn_model(data, nhid=16, dropout=0.5, lr=0.01, weight_decay=5e-4):
    model = GCN(data, nhid, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
