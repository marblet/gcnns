import torch


def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def get_degree(edge_list):
    row, col = edge_list
    deg = torch.bincount(row)
    return deg


def normalize_adj(edge_list):
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj


def preprocess_features(features):
    rowsum = features.sum(dim=1, keepdim=True)
    rowsum[rowsum == 0] = 1
    features = features / rowsum
    return features
