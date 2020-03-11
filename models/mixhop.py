import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.modules.module import Module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MixHop(nn.Module):
    def __init__(self, data, nhid, dropout):
        super(MixHop, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gcs1 = [MixHopConv(nfeat, nhid, data, hop=i) for i in range(0, 3)]
        for i, gc in enumerate(self.gcs1):
            self.add_module('mixhop1_{}'.format(i), gc)
        self.gcs2 = [MixHopConv(nhid*3, nclass, data, hop=i) for i in range(1, 2)]
        for i, gc in enumerate(self.gcs2):
            self.add_module('mixhop2_{}'.format(i), gc)
        self.dropout = dropout

    def reset_parameters(self):
        for gc in self.gcs1:
            gc.reset_parameters()
        for gc in self.gcs2:
            gc.reset_parameters()

    def forward(self, data):
        x = data.features
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([gc(x) for gc in self.gcs1], dim=1)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([gc(x) for gc in self.gcs2], dim=1)
        return F.log_softmax(x, dim=1)


class MixHopConv(Module):
    def __init__(self, in_features, out_features, data, hop, bias=True):
        super(MixHopConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        if hop > 0:
            self.adj = torch.pow(data.adj, hop)
        else:
            num_nodes = data.features.size(0)
            indices = torch.tensor([[i, i] for i in range(num_nodes)]).t()
            values = torch.tensor([1.0 for _ in range(num_nodes)])
            self.adj = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes)).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def forward(self, x):
        x = self.fc(x)
        x = torch.spmm(self.adj, x)
        return x


def create_mixhop_model(data, nhid=16, dropout=0.5):
    model = MixHop(data, nhid, dropout)
    return model