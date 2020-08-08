import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch

from utils import add_self_loops, normalize_adj


class Data(object):
    def __init__(self, adj, edge_list, features, labels, train_mask, val_mask, test_mask):
        self.adj = adj
        self.edge_list = edge_list
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_features = features.size(1)
        self.num_classes = int(torch.max(labels)) + 1

    def to(self, device):
        self.adj = self.adj.to(device)
        self.edge_list = self.edge_list.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)


def load_data(dataset_str, ntrain=20, seed=None):
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        data = load_planetoid_data(dataset_str)
    elif dataset_str == "wiki":
        data = load_wiki_data(ntrain, seed)
    elif dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        data = load_geom_data(dataset_str, ntrain, seed)
    else:
        data = load_npz_data(dataset_str, ntrain, seed)
    return data


def load_planetoid_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("data/planetoid/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.Tensor(out))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1))
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    edge_list = adj_list_from_dict(graph)
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj(edge_list)

    train_mask = index_to_mask(train_idx, labels.shape[0])
    val_mask = index_to_mask(val_idx, labels.shape[0])
    test_mask = index_to_mask(test_idx, labels.shape[0])

    data = Data(adj, edge_list, features, labels, train_mask, val_mask, test_mask)

    return data


def load_wiki_data(ntrain, seed):
    # generate feature matrix
    sp_feat = torch.tensor(np.loadtxt('data/wiki/tfidf.txt')).t()
    indices = sp_feat[:2].long()
    values = sp_feat[2].float()
    features = torch.sparse.FloatTensor(indices, values).to_dense()

    # generate edge list and adj matrix
    edge_list = torch.tensor(np.loadtxt('data/wiki/graph.txt')).long().t()
    edge_list_rev = torch.stack([edge_list[1], edge_list[0]])
    edge_list = torch.cat([edge_list, edge_list_rev], dim=1)
    edge_list = add_self_loops(edge_list, int(edge_list.max() + 1))
    adj = normalize_adj(edge_list)

    # generate labels and masks
    labels = torch.tensor(np.loadtxt('data/wiki/group.txt')).long().t()[1] - 1
    train_mask, val_mask, test_mask = split_data(labels, ntrain, 500, seed)

    data = Data(adj, edge_list, features, labels, train_mask, val_mask, test_mask)
    return data


def load_npz_data(dataset_str, ntrain, seed):
    with np.load('data/npz/' + dataset_str + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)
    if 'attr_data' in loader:
        feature_mat = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                    shape=loader['attr_shape']).todense()
    elif 'attr_matrix' in loader:
        feature_mat = loader['attr_matrix']
    else:
        feature_mat = None
    features = torch.tensor(feature_mat)

    adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                            shape=loader['adj_shape']).tocoo()
    edges = [(u, v) for u, v in zip(adj_mat.row.tolist(), adj_mat.col.tolist())]
    G = nx.Graph()
    G.add_nodes_from(list(range(features.size(0))))
    G.add_edges_from(edges)
    print(G.number_of_nodes())
    print(G.number_of_edges())
    edges = torch.tensor([[u, v] for u, v in G.edges()]).t()
    edge_list = torch.cat([edges, torch.stack([edges[1], edges[0]])], dim=1)
    edge_list = add_self_loops(edge_list, loader['adj_shape'][0])
    adj = normalize_adj(edge_list)

    if 'labels_data' in loader:
        labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                               shape=loader['labels_shape']).todense()
    elif 'labels' in loader:
        labels = loader['labels']
    else:
        labels = None
    labels = torch.tensor(labels).long()
    train_mask, val_mask, test_mask = split_data(labels, ntrain, 500, seed)

    data = Data(adj, edge_list, features, labels, train_mask, val_mask, test_mask)
    return data


def load_geom_data(dataset_str, ntrain, seed):
    # Feature and Label preprocessing
    with open('data/geom_data/{}/out1_node_feature_label.txt'.format(dataset_str)) as f:
        feature_labels = f.readlines()
    feat_list = []
    label_list = []
    for fl in feature_labels[1:]:
        id, feat, lab = fl.split('\t')
        feat = list(map(int, feat.split(',')))
        feat_list.append(feat)
        label_list.append(int(lab))
    features = torch.FloatTensor(feat_list)
    labels = torch.tensor(label_list).long()

    # Graph preprocessing
    with open('data/geom_data/{}/out1_graph_edges.txt'.format(dataset_str)) as f:
        edges = f.readlines()
    edge_pairs = []
    G = nx.Graph()
    for e in edges[1:]:
        u, v = map(int, e.split('\t'))
        edge_pairs.append((u, v))
    G.add_edges_from(edge_pairs)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    edge_list = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj(edge_list)

    train_mask, val_mask, test_mask = split_data(labels, ntrain, ntrain * 5, seed)

    data = Data(adj, edge_list, features, labels, train_mask, val_mask, test_mask)
    return data


def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def split_data(labels, n_train_per_class, n_val, seed):
    np.random.seed(seed)
    n_class = int(torch.max(labels)) + 1
    train_idx = np.array([], dtype=np.int64)
    remains = np.array([], dtype=np.int64)
    for c in range(n_class):
        candidate = torch.nonzero(labels == c).T.numpy()[0]
        np.random.shuffle(candidate)
        train_idx = np.concatenate([train_idx, candidate[:n_train_per_class]])
        remains = np.concatenate([remains, candidate[n_train_per_class:]])
    np.random.shuffle(remains)
    val_idx = remains[:n_val]
    test_idx = remains[n_val:]
    train_mask = index_to_mask(train_idx, labels.size(0))
    val_mask = index_to_mask(val_idx, labels.size(0))
    test_mask = index_to_mask(test_idx, labels.size(0))
    return train_mask, val_mask, test_mask
