import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class APPNP(nn.Module):
    def __init__(self, data, nhid, dropout, alpha, K):
        super(APPNP, self).__init__()
        self.fc1 = nn.Linear(data.num_features, nhid)
        self.fc2 = nn.Linear(nhid, data.num_classes)
        self.dropout = dropout
        self.prop = APPNP_prop(alpha, K)
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


class APPNP_prop(nn.Module):
    def __init__(self, alpha, K):
        super(APPNP_prop, self).__init__()
        self.alpha = alpha
        self.K = K

    def forward(self, x, adj):
        h = x
        for _ in range(self.K):
            x = (1 - self.alpha) * torch.spmm(adj, x) + self.alpha * h
        return x


def create_appnp_model(data, nhid=64, dropout=0.5, alpha=0.1, K=10, lr=0.01, weight_decay=5e-4):
    model = APPNP(data, nhid, dropout, alpha, K)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
