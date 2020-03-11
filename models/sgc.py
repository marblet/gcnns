import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class SGC(nn.Module):
    def __init__(self, data, K=2):
        super(SGC, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        x = data.features
        adj = data.adj
        self.fc = nn.Linear(nfeat, nclass)
        processed_x = x.clone()
        for _ in range(K):
            processed_x = torch.spmm(adj, processed_x)
        self.processed_x = processed_x

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, data):
        x = self.fc(self.processed_x)
        return F.log_softmax(x, dim=1)


def create_sgc_model(data):
    model = SGC(data)
    return model
