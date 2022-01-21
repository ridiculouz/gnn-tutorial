import numpy as np
import scipy.sparse as sp
import torch
from torch._C import dtype

def load_data(dataset="cora"):
    # return adj, features, labels, idx_train, idx_val, idx_test
    print("Loading data from {} dataset...".format(dataset))
    # load features
    features = np.genfromtxt("../data/{}/attributes".format(dataset), dtype=np.float32)
    features = sp.csr_matrix(features, dtype=np.float32)
    labels = np.genfromtxt("../data/{}/labels".format(dataset), dtype=np.int32)
    labels = np.array(labels[:, -1], dtype=np.int32)

    # build graph
    edges = np.genfromtxt("../data/{}/edges".format(dataset), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # normalize
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = np.genfromtxt("../data/{}/train_nodes".format(dataset), dtype=np.int32)
    idx_val = np.genfromtxt("../data/{}/val_nodes".format(dataset), dtype=np.int32)
    idx_test = np.genfromtxt("../data/{}/test_nodes".format(dataset), dtype=np.int32)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx   # D^-1A

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

if __name__ == "__main__":
    load_data()
    # A = sp.csr_matrix([[1.0, 0.0, 3], [2, 0, 0], [0, 1, 0]])
    # print(A)