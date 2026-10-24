"""B2 - GNN architectures for 12-task Tox21 prediction.

All models share one interface:
    model = build_model(name, ...)        # name in {"gcn", "mpnn", "attentivefp"}
    logits = model(batch)                 # batch: PyG Batch -> [B, 12] raw logits (no sigmoid)

Message passing in one sentence: each layer lets every atom update its vector from its bonded
neighbours, so after L layers an atom "sees" its L-hop neighbourhood. A readout (pooling) then
turns the atom vectors of a molecule into one molecule vector, and a linear head gives 12 logits.

Over-smoothing: stacking many layers makes atom vectors of a connected molecule converge to
similar values, losing local detail. That is why the number of layers is an ablation (G).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (AttentiveFP, GCNConv, GINEConv, global_add_pool,
                                global_max_pool, global_mean_pool)

from tox.constants import TASKS
from tox.featurize_graph import ATOM_DIM, BOND_DIM

POOLS = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}


def _bond_input(batch, use_bond_features: bool) -> torch.Tensor:
    """Bond features, or zeros of the same shape for the 'bond features off' ablation
    (keeps the architecture identical so only the information changes)."""
    return batch.edge_attr if use_bond_features else torch.zeros_like(batch.edge_attr)


class _Head(nn.Module):
    def __init__(self, hidden: int, n_tasks: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_tasks))

    def forward(self, h):
        return self.net(h)


class GCN(nn.Module):
    """Kipf & Welling GCN: neighbour average with symmetric degree normalisation.
    Atom features only - GCNConv has no place for bond features, so use_bond_features is ignored."""

    def __init__(self, in_dim=ATOM_DIM, hidden=128, n_layers=3, pooling="mean",
                 dropout=0.2, n_tasks=len(TASKS), **_):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(n_layers))
        self.pool, self.dropout = POOLS[pooling], dropout
        self.head = _Head(hidden, n_tasks, dropout)

    def forward(self, batch):
        h = batch.x
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.relu(norm(conv(h, batch.edge_index))), self.dropout, self.training)
        return self.head(self.pool(h, batch.batch))


class MPNN(nn.Module):
    """Message passing with bond features via GINEConv (Hu et al. 2020):
    h_i' = MLP((1+eps) h_i + sum_j ReLU(h_j + W e_ij)). Residual connections after layer 1."""

    def __init__(self, in_dim=ATOM_DIM, edge_dim=BOND_DIM, hidden=128, n_layers=3,
                 pooling="mean", dropout=0.2, use_bond_features=True,
                 n_tasks=len(TASKS), **_):
        super().__init__()
        self.use_bond_features = use_bond_features
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                     edge_dim=edge_dim, train_eps=True) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(n_layers))
        self.pool, self.dropout = POOLS[pooling], dropout
        self.head = _Head(hidden, n_tasks, dropout)

    def forward(self, batch):
        e = _bond_input(batch, self.use_bond_features)
        h = self.embed(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = F.relu(norm(conv(h, batch.edge_index, e)))
            h = F.dropout(h + h_new, self.dropout, self.training)    # residual
        return self.head(self.pool(h, batch.batch))


class AttentiveFPNet(nn.Module):
    """Xiong et al. 2020 AttentiveFP: attention-weighted message passing on atoms, then a
    GRU-based attention readout over `num_timesteps` steps (its own pooling; `pooling` ignored)."""

    def __init__(self, in_dim=ATOM_DIM, edge_dim=BOND_DIM, hidden=128, n_layers=3,
                 dropout=0.2, use_bond_features=True, num_timesteps=2,
                 n_tasks=len(TASKS), **_):
        super().__init__()
        self.use_bond_features = use_bond_features
        self.net = AttentiveFP(in_dim, hidden, n_tasks, edge_dim, n_layers, num_timesteps, dropout)

    def forward(self, batch):
        e = _bond_input(batch, self.use_bond_features)
        return self.net(batch.x, batch.edge_index, e, batch.batch)


MODELS = {"gcn": GCN, "mpnn": MPNN, "attentivefp": AttentiveFPNet}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)
