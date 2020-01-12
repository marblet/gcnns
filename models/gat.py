import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAT(nn.Module):
    def __init__(self, data, nhid, nheads, alpha, dropout):
        super(GAT, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc1 = [GATConv(nfeat, nhid, alpha, dropout) for _ in range(nheads)]
        for i, gc in enumerate(self.gc1):
            self.add_module('attention_{}'.format(i), gc)
        self.gc2 = GATConv(nhid * nheads, nclass, alpha, dropout)
        self.dropout = dropout

    def reset_parameters(self):
        for gc in self.gc1:
            gc.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_list = data.features, data.edge_list
        x = torch.cat([gc(x, edge_list) for gc in self.gc1], dim=1)
        x = F.elu(x)
        x = self.gc2(x, edge_list)
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


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, alpha, dropout, bias=True):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.att = nn.Conv1d(1, 1, 2 * out_features, bias=False)

        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att.weight, gain=1.414)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def forward(self, x, edge_list):
        N = x.size()[0]

        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.fc(x)

        edge_h = torch.cat((h[edge_list[0, :], :], h[edge_list[1, :], :]), dim=1)
        edge_e = torch.unsqueeze(edge_h, 1)
        edge_e = torch.squeeze(self.att(edge_e))
        edge_e = torch.exp(self.leakyrelu(edge_e))

        e_rowsum = self.special_spmm(edge_list, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=device))

        edge_e = F.dropout(edge_e, p=self.dropout, training=self.training)
        h_prime = self.special_spmm(edge_list, edge_e, torch.Size([N, N]), h)
        h_prime = h_prime.div(e_rowsum)

        return h_prime


def create_gat_model(data, nhid=8, nhead=8, alpha=0.2, dropout=0.6, lr=0.005, weight_decay=5e-4):
    model = GAT(data, nhid, nhead, alpha, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
