import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class GFNN(nn.Module):
    def __init__(self, data, nhid, dropout, K=2):
        super(GFNN, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.prelu = nn.PReLU()
        processed_x = data.features.clone()
        for _ in range(K):
            processed_x = torch.spmm(data.adj, processed_x)
        self.processed_x = processed_x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data):
        x = self.fc1(self.processed_x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prelu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create_gfnn_model(data, nhid=32, dropout=0.5):
    model = GFNN(data, nhid, dropout)
    return model
