import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models.gcn import GCNConv


class VGAE(nn.Module):
    def __init__(self, data, nhid, nlat, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GCNConv(data.num_features, nhid)
        self.gc2_mean = GCNConv(nhid, nlat)
        self.gc2_logvar = GCNConv(nhid, nlat)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2_mean.reset_parameters()
        self.gc2_logvar.reset_parameters()

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        hid1 = self.gc1(x, adj)
        hid1 = F.dropout(hid1, p=self.dropout, training=self.training)
        mu = self.gc2_mean(hid1, adj)
        logvar = self.gc2_logvar(hid1, adj)
        x = self.reparameterize(mu, logvar)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.mm(x, x.t())
        return x


def create_vgae_model(data, nhid=32, nlat=16, dropout=0., lr=0.01, weight_decay=0.):
    model = VGAE(data, nhid, nlat, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer