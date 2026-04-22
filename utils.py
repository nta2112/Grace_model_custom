import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
import pprint
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def task_generator(id_by_class, class_list, n_way, k_shot, m_query):

    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected



def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M
import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features, labels, temperature=0.5):
    device = features.device
    N = features.size(0)
    sim_matrix = torch.matmul(features, features.T) / temperature  # [N, N]

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    mask = mask - torch.eye(N, device=device)

    exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(N, device=device))
    sum_exp = torch.sum(exp_sim, dim=1, keepdim=True) + 1e-8

    log_prob = sim_matrix - torch.log(sum_exp)

    loss_per_sample = - torch.sum(mask * log_prob, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
    loss = torch.mean(loss_per_sample)
    return loss


def new_loss_function_with_contrast(output, labels_new, gate_weights, low_feature, high_feature,
                                    support_features, support_labels, query_features, query_labels,
                                    lambda_cls, lambda_gate, lambda_orth, lambda_contrast, temperature):

    loss_cls = F.nll_loss(output, labels_new)
    entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1)
    loss_gate = -lambda_gate * torch.mean(entropy)


    cosine_sim = torch.sum(low_feature * high_feature, dim=1)
    loss_orth = lambda_orth * torch.mean(cosine_sim ** 2)


    contrast_features = torch.cat([support_features, query_features], dim=0)
    contrast_labels = torch.cat([support_labels, query_labels], dim=0)

    contrast_norm = F.normalize(contrast_features, p=2, dim=1)
    loss_contrast = supervised_contrastive_loss(contrast_norm, contrast_labels, temperature)


    loss_total = lambda_cls*loss_cls + loss_gate + loss_orth + lambda_contrast * loss_contrast

    return loss_total


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch


hidden = {
    'CoraFull': 32,
    'dblp': 32,
    'ogbn-arxiv': 32,
    'Cora': 512,
    'CiteSeer': 32,
    'Amazon-Computer': 32,
    'Coauthor-CS': 32,
    'tlu': 32
}
dropt = {
    'CoraFull': 0,
    'dblp': 0,
    'ogbn-arxiv': 0,
    'Cora': 0.6,
    'CiteSeer': 0,
    'Amazon-Computer': 0,
    'Coauthor-CS': 0,
    'tlu': 0
}
patience = {
    'CoraFull': 10,
    'dblp': 10,
    'ogbn-arxiv': 10,
    'Cora': 5,
    'CiteSeer': 10,
    'Amazon-Computer': 10,
    'Coauthor-CS': 10,
    'tlu': 10
}
lr = {
    'CoraFull': 0.001,
    'dblp': 0.001,
    'ogbn-arxiv': 0.001,
    'Cora': 0.0005,
    'CiteSeer': 0.001,
    'Amazon-Computer': 0.001,
    'Coauthor-CS': 0.001,
    'tlu': 0.001
}
gate = {
    'CoraFull': 0.1,
    'dblp': 0.1,
    'ogbn-arxiv': 0.1,
    'Cora': 0.05,
    'CiteSeer': 0.1,
    'Amazon-Computer': 0.1,
    'Coauthor-CS': 0.1,
    'tlu': 0.1
}
contrast = {
    'CoraFull': 1.,
    'dblp': 1.,
    'ogbn-arxiv': 1.,
    'Cora': 0.99,
    'CiteSeer': 1.,
    'Amazon-Computer': 1.,
    'Coauthor-CS': 1.,
    'tlu': 1.
}
