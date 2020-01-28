import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAT(nn.Module):
    def __init__(self, data, nhid, nhead, alpha, dropout):
        """Dense version of GAT."""
        nfeat, nclass = data.num_features, data.num_classes
        super(GAT, self).__init__()

        self.attentions = [GATConv(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nhead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATConv(nhid * nhead, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.reset_parameters()

    def reset_parameters(self):
        for att in self.attentions:
            att.reset_parameters()
        self.out_att.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        adj = adj.to_dense()
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)


class GATConv(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, bias=True):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.fc(x)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class SpGAT(nn.Module):
    def __init__(self, data, nhid, nhead, alpha, dropout):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.attentions = [SpGATConv(nfeat,
                                     nhid,
                                     dropout=dropout,
                                     alpha=alpha,
                                     concat=True) for _ in range(nhead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGATConv(nhid * nhead,
                                 nclass,
                                 dropout=dropout,
                                 alpha=alpha,
                                 concat=False)

    def reset_parameters(self):
        for att in self.attentions:
            att.reset_parameters()
        self.out_att.reset_parameters()

    def forward(self, data):
        x, edge = data.features, data.edge_list
        x = torch.cat([att(x, edge) for att in self.attentions], dim=1)
        x = self.out_att(x, edge)
        return F.log_softmax(x, dim=1)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGATConv(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, bias=True):
        super(SpGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))

        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge):
        N = x.size()[0]

        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.fc(x)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=device))
        # e_rowsum: N x 1

        edge_e = F.dropout(edge_e, p=self.dropout, training=self.training)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


def create_gat_model(data, nhid=8, nhead=8, alpha=0.2, dropout=0.6, lr=0.005, weight_decay=5e-4):
    model = GAT(data, nhid, nhead, alpha, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
