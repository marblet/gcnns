import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class APPNP(nn.Module):
    def __init__(self, data, nhid=64, dropout=0.5, alpha=0.1, K=10):
        super(APPNP, self).__init__()
        self.fc1 = nn.Linear(data.num_features, nhid)
        self.fc2 = nn.Linear(nhid, data.num_classes)
        self.dropout = dropout
        self.prop = APPNPprop(alpha, K, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = self.prop(x, adj)
        return F.log_softmax(x, dim=1)


class APPNPprop(nn.Module):
    def __init__(self, alpha, K, dropout):
        super(APPNPprop, self).__init__()
        self.alpha = alpha
        self.K = K
        self.dropout = dropout

    def forward(self, x, adj):
        h = x
        edge_list = adj._indices()
        values = adj._values()
        one_mat = torch.ones_like(values)
        for _ in range(self.K):
            dropped_values = values * F.dropout(one_mat, p=self.dropout, training=self.training)
            dropped_adj = torch.sparse.FloatTensor(edge_list, dropped_values)
            x = (1 - self.alpha) * torch.spmm(dropped_adj, x) + self.alpha * h
        return x
