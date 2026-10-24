"""B2 - GNN architectures for 12-task Tox21 prediction.

All models share one interface:
    model = build_model(name, ...)        # name in {"gcn", "mpnn", "afp"} (plan tokens)
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
from torch_geometric.nn import AttentiveFP, GCNConv, GINEConv, global_max_pool, global_mean_pool

from tox.constants import TASKS
from tox.featurize_graph import ATOM_DIM, BOND_DIM

POOLING_MODES = ("mean", "max", "meanmax")      # plan B2/B5: mean, max, and mean||max (the default)


def _check_pooling(pooling: str) -> str:
    if pooling not in POOLING_MODES:
        raise ValueError(f"unknown pooling {pooling!r}; choose from {POOLING_MODES}")
    return pooling


def readout(h: torch.Tensor, batch_index: torch.Tensor, pooling: str) -> torch.Tensor:
    """Graph readout: turn atom vectors [N, d] into molecule vectors [B, d] or [B, 2d].

    'meanmax' concatenates the mean over atoms (overall composition) with the max over atoms
    (strongest local signal, e.g. one toxicophore), giving 2d dimensions.
    """
    if pooling == "mean":
        return global_mean_pool(h, batch_index)
    if pooling == "max":
        return global_max_pool(h, batch_index)
    return torch.cat([global_mean_pool(h, batch_index), global_max_pool(h, batch_index)], dim=1)


def pooled_dim(hidden: int, pooling: str) -> int:
    return 2 * hidden if pooling == "meanmax" else hidden


def _bond_input(batch, use_bond_features: bool) -> torch.Tensor:
    """Bond features, or zeros of the same shape for the 'bond features off' ablation
    (keeps the architecture identical so only the information changes)."""
    return batch.edge_attr if use_bond_features else torch.zeros_like(batch.edge_attr)


class _Head(nn.Module):
    """Prediction MLP: pooled vector -> hidden -> n_tasks. With the defaults (hidden 128,
    mean||max pooling) this is the plan's 256 -> 128 -> 12."""

    def __init__(self, in_dim: int, hidden: int, n_tasks: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_tasks))

    def forward(self, h):
        return self.net(h)


class GCN(nn.Module):
    """Kipf & Welling GCN: neighbour average with symmetric degree normalisation.
    Atom features only - GCNConv has no place for bond features, so use_bond_features is ignored."""

    def __init__(self, in_dim=ATOM_DIM, hidden=128, n_layers=3, pooling="meanmax",
                 dropout=0.2, n_tasks=len(TASKS), **_):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(n_layers))
        self.pooling, self.dropout = _check_pooling(pooling), dropout
        self.head = _Head(pooled_dim(hidden, pooling), hidden, n_tasks, dropout)

    def forward(self, batch):
        h = batch.x
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.relu(norm(conv(h, batch.edge_index))), self.dropout, self.training)
        return self.head(readout(h, batch.batch, self.pooling))


class MPNN(nn.Module):
    """Plan B2: n_layers x GINEConv (each with an MLP and bond features via edge_dim), the same
    mean||max readout and 256 -> 128 -> 12 head as the GCN.

    GINE (Hu et al. 2020): h_i' = MLP((1+eps) h_i + sum_j ReLU(h_j + W e_ij)); the bond features
    e_ij are projected by a learned linear map W, which is how this model uses bonds and GCN does
    not. Each layer is followed by BatchNorm + ReLU + dropout, mirroring the GCN block (the plan
    gives that block explicitly for gcn only; reusing it here is our assumption). No input
    embedding and no residual connections: the plan lists neither.
    """

    def __init__(self, in_dim=ATOM_DIM, edge_dim=BOND_DIM, hidden=128, n_layers=3,
                 pooling="meanmax", dropout=0.2, use_bond_features=True,
                 n_tasks=len(TASKS), **_):
        super().__init__()
        self.use_bond_features = use_bond_features
        dims = [in_dim] + [hidden] * n_layers
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(a, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                     edge_dim=edge_dim, train_eps=True) for a in dims[:-1])
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(n_layers))
        self.pooling, self.dropout = _check_pooling(pooling), dropout
        self.head = _Head(pooled_dim(hidden, pooling), hidden, n_tasks, dropout)

    def forward(self, batch):
        e = _bond_input(batch, self.use_bond_features)
        h = batch.x
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.relu(norm(conv(h, batch.edge_index, e))), self.dropout, self.training)
        return self.head(readout(h, batch.batch, self.pooling))


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


MODELS = {"gcn": GCN, "mpnn": MPNN, "afp": AttentiveFPNet}     # keys are the C3 filename tokens (plan p. 5)
ALIASES = {"attentivefp": "afp"}                                 # earlier name, still accepted


def canonical_name(name: str) -> str:
    """Map an accepted alias to the plan's model token (used in C3/C4 file names)."""
    return ALIASES.get(name, name)


def build_model(name: str, **kwargs) -> nn.Module:
    name = canonical_name(name)
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)
