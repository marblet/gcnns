import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class ELGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(ELGCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.W1 = torch.rand(size=(nfeat, nhid))
        self.fc2 = nn.Linear(nhid, nclass)

    def reset_parameters(self):
        self.W1 = torch.rand(size=(self.nfeat, self.nhid))
        self.fc2.reset_parameters()

    def forward(self, data):
        x = data.features
        adj = data.adj
        x = torch.spmm(adj, x)
        x = torch.matmul(x, self.W1)
        x = F.relu(x)
        x = torch.spmm(adj, x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create_elgcn_model(data, nhid=16, dropout=0.5, lr=0.01, weight_decay=5e-4):
    model = ELGCN(data.num_features, nhid, data.num_classes)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
