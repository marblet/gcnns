import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class SGC(nn.Module):
    def __init__(self, nfeat, nclass, x, adj, K=2):
        super(SGC, self).__init__()
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


def create_sgc_model(data, lr=0.2, weight_decay=3e-5):
    model = SGC(data.num_features, data.num_classes, data.features, data.adj)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
