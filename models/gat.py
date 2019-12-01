import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.optim import Adam


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, nhead, alpha, dropout):
        super(GAT, self).__init__()
        self.gc1 = GATConv(nfeat, nhid, nhead, alpha, dropout)
        self.gc2 = GATConv(nhid * nhead, nout, 1, alpha, dropout)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_list = data.features, data.edge_list
        x = F.elu(self.gc1(x, edge_list))
        x = self.gc2(x, edge_list)
        return F.log_softmax(x, dim=1)


class GATConv(Module):
    def __init__(self, in_features, out_features, nhead, alpha, dropout, bias=True, concat=True):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.weight = Parameter(torch.Tensor(nhead, in_features, out_features))
        self.att = Parameter(torch.Tensor(nhead, 2 * out_features, 1))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(out_features * nhead))
        elif bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, edge_list):
        source, target = edge_list
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.matmul(x, self.weight)
        wh_cat = torch.cat((x[:, source], x[:, target]), dim=2)
        attention = F.leaky_relu(torch.bmm(wh_cat, self.att), self.alpha).view(self.nhead, -1)
        adj = torch.stack([torch.sparse.FloatTensor(edge_list, a) for a in attention]).to_dense()
        print(adj.size())
        adj = F.softmax(adj, dim=1)
        x = torch.bmm(adj, x)
        if self.concat:
            x = torch.cat([a for a in x], dim=1)
        if self.bias is not None:
            return x + self.bias
        else:
            return x


def create_gat_model(data, nhid=16, nhead=8, alpha=0.1, dropout=0.6, lr=0.01, weight_decay=5e-4):
    model = GAT(data.num_features, nhid, data.num_classes, nhead, alpha, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
