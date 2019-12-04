import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.modules.module import Module

from utils import get_degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphSAGE(nn.Module):
    def __init__(self, data, nhid, dropout, aggr='mean'):
        super(GraphSAGE, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc1 = GraphSAGELayer(nfeat, nhid, data)
        self.gc2 = GraphSAGELayer(nhid, nclass, data)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x = data.features
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)


class MeanAggregator(object):
    def __init__(self, data):
        self.edge_list = data.edge_list
        self.degree = get_degree(data.edge_list).float().view(-1, 1).to(device)

    def aggregate(self, x):
        source, _ = self.edge_list
        h = torch.zeros(x.size(), device=device)
        h.index_add_(0, source, x[source])
        h = h / self.degree
        return h


class GraphSAGELayer(Module):
    def __init__(self, in_features, out_features, data, bias=True):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.aggr = MeanAggregator(data)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def forward(self, x):
        x = self.fc(x)
        x = self.aggr.aggregate(x)
        return x


def create_graphsage_model(data, nhid=16, dropout=0.5, lr=0.01, weight_decay=5e-4):
    model = GraphSAGE(data, nhid, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
