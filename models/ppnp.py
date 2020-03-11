import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_ppr_exact(adj, alpha):
    A_hat = adj.to_dense()
    A_inner = torch.eye(A_hat.size(0)) - (1 - alpha) * A_hat
    return alpha * np.linalg.inv(A_inner.numpy())


class PPNP(nn.Module):
    def __init__(self, data, nhid, dropout, alpha):
        super(PPNP, self).__init__()
        self.fc1 = nn.Linear(data.num_features, nhid)
        self.fc2 = nn.Linear(nhid, data.num_classes)
        self.dropout = dropout
        self.prop_adj = torch.tensor(calc_ppr_exact(data.adj, alpha)).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data):
        x = data.features
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        prop_mat = F.dropout(self.prop_adj, p=self.dropout, training=self.training)
        x = torch.matmul(prop_mat, x)
        return F.log_softmax(x, dim=1)


def create_ppnp_model(data, nhid=64, dropout=0.5, alpha=0.1):
    model = PPNP(data, nhid, dropout, alpha)
    return model
