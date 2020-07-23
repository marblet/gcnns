import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils import get_degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MaskedGCNSym(nn.Module):
    def __init__(self, data, dropout=0.5):
        super(MaskedGCNSym, self).__init__()
        self.fc = nn.Linear(data.num_features, data.num_classes, bias=False)
        self.degree = get_degree(data.edge_list).float().to(device)
        self.dense_adj = data.adj.to_dense().to(device)
        self.sigma = Parameter(torch.FloatTensor(data.num_features))
        self.dropout = dropout

    def reset_parameters(self):
        self.fc.reset_parameters()
        self.sigma.data = torch.ones_like(self.sigma) * 100

    def calc_edge_weight(self, x, edge_list):
        source, target = edge_list
        diff = torch.pow((x[source] - x[target]), 2)
        diff = diff / (self.sigma * self.sigma)
        edge_weight = torch.exp(- torch.sum(diff, dim=1))
        return edge_weight

    def generate_mask(self, x, edge_list, edge_weight):
        source, target = edge_list
        diff = torch.pow((x[source] - x[target]), 2)
        diff = edge_weight.unsqueeze(1) * diff
        mask = torch.zeros(x.size(), device=device)
        mask.index_add_(0, source, diff)
        mask = mask / (self.sigma * self.sigma)
        mask = torch.exp(- mask / self.degree.unsqueeze(1))
        return mask

    def update(self, x, edge_list, edge_weight):
        feature_mask = self.generate_mask(x, edge_list, edge_weight)
        x = x * feature_mask
        source, target = edge_list
        weight_sum = torch.zeros(x.size(0), device=device)
        weight_sum.index_add_(0, source, edge_weight)
        edge_weight = edge_weight / weight_sum[source]
        h = torch.zeros_like(x)
        h.index_add_(0, source, edge_weight.unsqueeze(1) * x[target])
        return h

    def forward(self, data):
        x, edge_list = data.features, data.edge_list
        edge_weight = self.calc_edge_weight(x, edge_list)
        x = self.update(x, edge_list, edge_weight)
        x = self.update(x, edge_list, edge_weight)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class MaskedGCNAsym(MaskedGCNSym):
    def __init__(self, data, dropout=0.5):
        super(MaskedGCNAsym, self).__init__(data, dropout)
        self.b = nn.Parameter(torch.FloatTensor(2 * data.num_features))

    def calc_edge_weight(self, x, edge_list):
        source, target = edge_list
        x_cat = torch.cat((x[source], x[target]), dim=1)
        edge_weight = torch.exp(torch.sum(self.b * x_cat, dim=1))
        return edge_weight
