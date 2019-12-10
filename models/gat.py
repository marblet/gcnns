import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.optim import Adam


class GAT(nn.Module):
    def __init__(self, data, nhid, nheads, alpha, dropout):
        super(GAT, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc1 = [GATConv(nfeat, nhid, alpha, dropout) for _ in range(nheads)]
        self.gc2 = GATConv(nhid * nheads, nclass, alpha, dropout)
        self.dropout = dropout

    def reset_parameters(self):
        for gc in self.gc1:
            gc.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_list = data.features, data.edge_list
        x = torch.cat([gc(x, edge_list) for gc in self.gc1], dim=1)
        x = F.elu(x)
        x = self.gc2(x, edge_list)
        return F.log_softmax(x, dim=1)


class GATConv(Module):
    def __init__(self, in_features, out_features, alpha, dropout, bias=True):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.att = nn.Conv1d(1, 1, 2 * out_features, bias=False)
        self.alpha = alpha
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att.weight, gain=1.414)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def forward(self, x, edge_list):
        source, target = edge_list
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        wh_cat = torch.cat((x[source], x[target]), dim=1)
        wh_cat = torch.unsqueeze(wh_cat, 1)
        attention = F.leaky_relu(self.att(wh_cat), negative_slope=self.alpha)
        attention = torch.squeeze(attention)
        adj = torch.sparse.FloatTensor(edge_list, attention).to_dense()
        adj[adj == 0] = -9e15
        adj = F.softmax(adj, dim=1)
        adj = F.dropout(adj, p=self.dropout, training=self.training)
        x = torch.matmul(adj, x)
        return x


def create_gat_model(data, nhid=8, nhead=8, alpha=0.1, dropout=0.6, lr=0.01, weight_decay=5e-4):
    model = GAT(data, nhid, nhead, alpha, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
