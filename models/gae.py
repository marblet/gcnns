import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models.gcn import GCNConv


class GAE(nn.Module):
    def __init__(self, data, nhid, nlat, dropout):
        super(GAE, self).__init__()
        self.gc1 = GCNConv(data.num_features, nhid)
        self.gc2 = GCNConv(nhid, nlat)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        x = self.gc2(x, adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.mm(x, x.t())
        return x


def create_gae_model(data, nhid=32, nlat=16, dropout=0., lr=0.01, weight_decay=0.):
    model = GAE(data, nhid, nlat, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
