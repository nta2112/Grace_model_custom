
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from layers import GraphConvolution


class LowPassExtractor(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(LowPassExtractor, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.bn1 = nn.BatchNorm1d(2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        low_feature = F.relu(self.bn1(self.gc1(x, adj)))
        low_feature = F.dropout(low_feature, self.dropout, training=self.training)
        low_feature = F.relu(self.bn2(self.gc2(low_feature, adj)))
        return low_feature


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_min

class Gating(nn.Module):
    def __init__(self, nfeat, nhid, num_expert=2, dropout=0.5, temperature=1.0):
        super(Gating, self).__init__()
        in_dim = 2 * nfeat + 1 + nfeat
        self.fc1 = nn.Linear(in_dim, nhid)
        self.norm1 = nn.LayerNorm(nhid)
        self.fc2 = nn.Linear(nhid, num_expert)
        self.dropout = dropout
        self.temperature = temperature
        self._x_cat = None
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, adj):

        if self._x_cat is None:
            N, D = x.size()

            AX = torch.spmm(adj, x)       # [N, D]
            delta1 = AX - x               # [N, D]

            deg = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1)  # [N, 1]

            x_mean = AX / (deg + 1e-8)    # [N, D]
            AX2 = torch.spmm(adj, x * x)
            var = AX2 / (deg + 1e-8) - x_mean * x_mean
            x_std = torch.sqrt(torch.clamp(var, min=0.0))             # [N, D]
            self._x_cat = torch.cat([
                x,         # [N, D]
                delta1,    # [N, D]
                x_std,
                deg# [N, D]
            ], dim=1)     # [N, in_dim]

        h = F.relu(self.norm1(self.fc1(self._x_cat)))
        h = F.dropout(h, self.dropout, training=self.training)
        logits = self.fc2(h)
        return F.softmax(logits / self.temperature, dim=1)


class LowPassExpert(nn.Module):

    def __init__(self):
        super(LowPassExpert, self).__init__()

    def forward(self, low_feature, adj=None):

        return low_feature

class SparseAttentionHighPass(nn.Module):


    def __init__(self, nfeat, nhid, dropout=0.5, alpha=0.2):
        super(SparseAttentionHighPass, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.q_linear = nn.Linear(nhid, nhid, bias=False)
        self.k_linear = nn.Linear(nhid, nhid, bias=False)
        self.v_linear = nn.Linear(nhid, nhid, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, residual, adj):

        N = residual.size(0)
        Q = self.q_linear(residual)  # [N, nhid]
        K = self.k_linear(residual)  # [N, nhid]
        V = self.v_linear(residual)  # [N, nhid]
        edge_index = adj._indices()  # [2, E]
        Q_i = Q[edge_index[0]]  # [E, nhid]
        K_j = K[edge_index[1]]  # [E, nhid]
        scores = torch.sum(Q_i * K_j, dim=1) / (self.nhid ** 0.5)  # [E]
        scores = self.leakyrelu(scores)
        exp_scores = torch.exp(scores)
        denom = torch_scatter.scatter_add(exp_scores, edge_index[0], dim=0, dim_size=N)
        attn_coeff = exp_scores / (denom[edge_index[0]] + 1e-16)
        attn_coeff = F.dropout(attn_coeff, self.dropout, training=self.training)
        high_feature = torch_scatter.scatter_add(
            attn_coeff.unsqueeze(1) * V[edge_index[1]],
            edge_index[0],
            dim=0,
            dim_size=N
        )
        return high_feature


class HighPassExpert(nn.Module):

    def __init__(self, nfeat, nhid, dropout=0.5, alpha=0.2, residual_balance=1.0):
        super(HighPassExpert, self).__init__()
        self.x_proj = nn.Linear(nfeat, nhid)
        self.attn_high = SparseAttentionHighPass(nfeat, nhid, dropout=dropout, alpha=alpha)
        self.residual_scale = nn.Parameter(torch.tensor(residual_balance, dtype=torch.float32))
        self.residual_norm = nn.LayerNorm(nhid)

    def forward(self, x, low_feature, adj):
        x_proj = self.x_proj(x)  # [N, nhid]
        residual = x_proj - low_feature  # [N, nhid]
        residual = self.residual_scale * residual
        residual = self.residual_norm(residual)
        high_feature = self.attn_high(residual, adj)
        return high_feature



class MOE(nn.Module):

    def __init__(self, nfeat, nhid, dropout=0.5, expert_types=["low", "high"], temperature=2):
        super(MOE, self).__init__()
        self.lowpass_extractor = LowPassExtractor(nfeat, nhid, dropout=dropout)
        num_expert = len(expert_types)
        self.gating = Gating(nfeat, nhid, num_expert=num_expert, dropout=dropout,temperature=temperature)
        self.experts = nn.ModuleList()
        for etype in expert_types:
            if etype == "low":
                self.experts.append(LowPassExpert())
            elif etype == "high":
                self.experts.append(HighPassExpert(nfeat, nhid, dropout=dropout))
        self.nhid = nhid
        self.lambda_corr = nn.Parameter(torch.tensor(0, dtype=torch.float32))
    def forward(self, x, adj):
        low_feature = self.lowpass_extractor(x, adj)  # [N, nhid]
        gate_weights = self.gating(x, adj)
        output = 0
        high_feature = None
        for i, expert in enumerate(self.experts):
            if i == 0:
                expert_output = expert(low_feature, adj)
            elif i == 1:
                expert_output = expert(x, low_feature, adj)
                high_feature = expert_output
            else:
                expert_output = expert(x, adj)
            output = output + gate_weights[:, i].unsqueeze(1) * expert_output
        return output, gate_weights, low_feature, high_feature

    def forward_with_correction(self, x, adj, support_idx, query_idx, n_way, k_shot,bandwidth):
        embeddings, gate_weights, low_feature, high_feature = self.forward(x, adj)
        z_dim = embeddings.size(1)
        support_embeddings = embeddings[support_idx].view(n_way, k_shot, z_dim)
        prototype_embeddings = support_embeddings.mean(dim=1)  # [n_way, d]
        query_embeddings = embeddings[query_idx]  # [n_query, d]
        corrected_prototypes = []
        lambda_corr_limited = 0.5 * (torch.tanh(self.lambda_corr) + 1)
        bandwidth = bandwidth
        temperature = 0.9
        for i in range(n_way):
            diff = query_embeddings - prototype_embeddings[i].unsqueeze(0)
            sq_dist = torch.sum(diff ** 2, dim=1)
            weights = torch.exp(-sq_dist / (2 * temperature * (bandwidth ** 2)))
            weights = weights / (torch.sum(weights) + 1e-8)
            correction = torch.sum(weights.unsqueeze(1) * diff, dim=0)
            new_proto = prototype_embeddings[i] + lambda_corr_limited * correction
            corrected_prototypes.append(new_proto)
        corrected_prototypes = torch.stack(corrected_prototypes)  # [n_way, d]
        return embeddings, gate_weights, low_feature, high_feature, corrected_prototypes

