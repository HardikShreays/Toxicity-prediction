"""Phase 2 hybrid models (plan p. 18-19): three variants from two trained parents.

Parents (per split + seed, always trained on THAT split's train rows only):
  * the frozen tuned MPNN graph encoder (results/gnn_params.json + the checkpoint
    results/gnn/mpnn__{split}__seed{s}.pt written by tox.train_gnn),
  * the Morgan fingerprint (results/processed/fp_morgan2048.npy, row-aligned with C1).

Variants (file tokens, the C3 model column):
  hyb_embxgb  frozen GNN: pooled readout embedding [n, pooled_dim] || Morgan fingerprint
              -> one XGBoost per endpoint (Arjun's variant, plan p. 18). The encoder is
              only USED here, never retrained, so no gradient touches the graph weights.
  hyb_stack   per-endpoint logistic regression on the validation probabilities of
              xgb_fpdesc and mpnn; fitted on val, applied to test (Arjun's variant).
              Never fitted on test (plan rule 2).
  hyb_fusion  the joint model (Hardik's variant): fingerprint through a small MLP branch,
              graph through the MPNN encoder branch (warm-started from the checkpoint of
              the same split+seed), concatenated before one shared head, trained end to
              end with the same masked BCE, Adam recipe and early stopping as the GNNs.

Leakage rules (plan p. 19): the encoder comes from the same split's train part; the
stacking meta-learner is fit on val only; every variant is reported against both parents
on the same splits and seeds.

The graph branch reuses the MPNN building blocks (GINEConv stack, mean||max readout) from
tox.models_gnn with identical attribute names (convs, norms), so a trained MPNN checkpoint
warm-starts the fusion encoder by name; incompatible keys (the old 12-logit head) are
skipped by `warm_start`. Assumption (the plan does not say): the hybrid graph branch is
the message-passing MPNN - the model the plan's B2 brief calls "the message-passing model
the brief asks for" - not GCN (no bond features) or AttentiveFP (no separable readout).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv

from tox.constants import CLEAN, HYBRID, TASKS
from tox.featurize_graph import ATOM_DIM, BOND_DIM
from tox.models_gnn import _Head, _check_pooling, pooled_dim, readout

FP_DIM = 2048                                    # Morgan r2/2048, the plan's fingerprint

ARCH_KEYS = ("hidden", "n_layers", "pooling", "dropout", "use_bond_features")


def load_mpnn_encoder(split: str, seed: int, cfg: dict, params_path=None):
    """The trained MPNN encoder for one (split, seed), from the GNN grid's saved weights.

    cfg is the FROZEN tuned config (tox.train_gnn.frozen_config); the checkpoint was trained
    on this split's train rows only (early-stopped on its val), so reusing it is the plan's
    rule 1, not leakage.
    """
    from tox.models_gnn import build_model        # local import: models_gnn stays torch-only
    from tox.train_gnn import GNN_RESULTS, set_seed

    ckpt = GNN_RESULTS / f"mpnn__{split}__seed{seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} is missing; run the GNN grid first (./run_gnn.sh)")
    set_seed(seed)
    model = build_model("mpnn", **{k: cfg[k] for k in ARCH_KEYS})
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


def with_fps(graphs, fp: np.ndarray, smiles: pd.Series | list[str]) -> list:
    """Copies of `graphs` carrying a `fp` uint8 tensor [1, 2048].

    The fingerprint of the molecule the graph came from, looked up in the C1-aligned cache
    by smiles. PyG concatenates [1, k] per-molecule attributes along dim 0, so a batch gets
    `batch.fp` of shape [B, 2048] - the same trick as the label attribute `y`. Original
    graphs are not mutated (they are shared between runs).
    """
    pos = pd.Series(range(len(smiles)), index=list(smiles))
    out = []
    for g in graphs:
        d = g.clone()
        d.fp = torch.as_tensor(fp[pos[g.smiles]], dtype=torch.uint8).reshape(1, FP_DIM)
        out.append(d)
    return out


class Fusion(torch.nn.Module):
    """hyb_fusion: graph branch (MPNN encoder -> readout) + fingerprint branch (2048 -> fp_hidden
    MLP), concatenated before a shared head, trained end to end with masked BCE.

    `use_fp` / `use_graph` exist for the ablation arms (plan p. 19: fusion minus the FP
    branch / minus the graph branch); exactly one must stay on. The head mirrors the GNN
    head (in -> hidden -> n_tasks with dropout); with the tuned 128-hidden mean||max encoder
    and fp_hidden 128 its input is 256 + 128 = 384.
    """

    def __init__(self, hidden: int = 128, n_layers: int = 3, pooling: str = "meanmax",
                 dropout: float = 0.2, use_bond_features: bool = True,
                 fp_dim: int = FP_DIM, fp_hidden: int = 128, use_fp: bool = True,
                 use_graph: bool = True, n_tasks: int = len(TASKS)):
        super().__init__()
        if not (use_fp or use_graph):
            raise ValueError("a hybrid needs at least one branch (use_fp / use_graph)")
        self.use_fp, self.use_graph = use_fp, use_graph
        if use_graph:
            # same block, attribute names and init as tox.models_gnn.MPNN, so the frozen
            # grid checkpoint warm-starts it by name (see warm_start)
            self.use_bond_features = use_bond_features
            dims = [ATOM_DIM] + [hidden] * n_layers
            self.convs = torch.nn.ModuleList(
                GINEConv(Sequential(Linear(a, hidden), ReLU(), Linear(hidden, hidden)),
                         edge_dim=BOND_DIM, train_eps=True) for a in dims[:-1])
            self.norms = torch.nn.ModuleList(torch.nn.BatchNorm1d(hidden) for _ in range(n_layers))
            self.pooling, self.dropout = _check_pooling(pooling), dropout
        if use_fp:
            self.fp_branch = Sequential(Linear(fp_dim, fp_hidden), ReLU(),
                                        torch.nn.Dropout(dropout))
        head_in = (pooled_dim(hidden, pooling) if use_graph else 0) + (fp_hidden if use_fp else 0)
        self.head = _Head(head_in, hidden, n_tasks, dropout)

    def forward(self, batch):
        parts = []
        if self.use_graph:
            e = batch.edge_attr if self.use_bond_features else torch.zeros_like(batch.edge_attr)
            h = batch.x
            for conv, norm in zip(self.convs, self.norms):
                h = F.dropout(F.relu(norm(conv(h, batch.edge_index, e))),
                              self.dropout, self.training)
            parts.append(readout(h, batch.batch, self.pooling))
        if self.use_fp:
            if not hasattr(batch, "fp"):
                raise RuntimeError("graphs have no fingerprints; build them with tox.hybrid.with_fps")
            parts.append(self.fp_branch(batch.fp.float()))
        return self.head(torch.cat(parts, dim=1))


def warm_start(model: Fusion, ckpt_path, log=print) -> int:
    """Copy every checkpoint weight that exists in `model` with the same shape.

    The MPNN checkpoint fills the graph branch completely; its 12-logit head and anything
    fp-branch-shaped is skipped and stays freshly initialised. Returns the number of tensors
    loaded (28 for a 3-layer MPNN: convs, norms, eps).
    """
    state = torch.load(ckpt_path, map_location="cpu")
    own = model.state_dict()
    match = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(match, strict=False)
    skipped = sorted(set(state) - set(match))
    if skipped:
        log(f"warm start: loaded {len(match)} tensors from {os.path.basename(str(ckpt_path))}, "
            f"skipped {len(skipped)} ({skipped[0]}, ...)")
    return len(match)


@torch.no_grad()
def encoder_embeddings(model, graphs, batch_size: int = 256) -> np.ndarray:
    """Pooled readout vectors [n, pooled_dim] of a trained MPNN, in the order of `graphs`.

    The frozen-embedding features for hyb_embxgb: the encoder is used purely as a feature
    extractor (model.eval(), no gradients), so training data never updates its weights.
    """
    from tox.models_gnn import _bond_input          # private but the exact bond handling of MPNN
    model.eval()
    out = []
    for b in DataLoader(graphs, batch_size=batch_size):
        h = b.x
        for conv, norm in zip(model.convs, model.norms):
            h = F.dropout(F.relu(norm(conv(h, b.edge_index, _bond_input(b, model.use_bond_features)))),
                          model.dropout, model.training)
        out.append(readout(h, b.batch, model.pooling))
    return torch.cat(out).numpy().astype(np.float64)


def stack_features(parent_a_path, parent_b_path, role: str, labels: pd.DataFrame
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) for the stacking meta-learner: the two parents' probabilities for one role.

    X rows follow the C3 file's row order (the same order in both parent files, since both
    were written from the same split); y comes from C1 by smiles, NaN kept.
    """
    a, b = pd.read_parquet(parent_a_path), pd.read_parquet(parent_b_path)
    assert list(a.smiles) == list(b.smiles), "parent C3 files do not share row order"
    a, b = a[a.role == role], b[a.role == role]
    X = np.column_stack([a[TASKS].to_numpy(dtype=float), b[TASKS].to_numpy(dtype=float)])
    y = labels.loc[a.smiles, TASKS].to_numpy(dtype=float)
    return X, y


def stack_one(X: dict, y: dict, seed: int) -> tuple[dict, list[str]]:
    """Per-endpoint logistic regression on the parents' probabilities, fit on val, applied to
    val and test (plan rule 2: never fitted on test).

    X[r] is [n, 24]: columns 0-11 are parent A's probabilities, 12-23 parent B's. An endpoint
    whose validation labels do not contain both classes has nothing to fit on: the
    meta-learner falls back to the mean of the two parents for it (returned, and logged by
    the caller). class_weight balances the rare positives, as everywhere else in this repo.
    """
    probs = {r: np.zeros((X[r].shape[0], len(TASKS))) for r in X}
    fallback = []
    for j in range(len(TASKS)):
        m = ~np.isnan(y["val"][:, j])
        if len(np.unique(y["val"][m, j])) < 2:
            fallback.append(TASKS[j])
            for r in probs:
                probs[r][:, j] = X[r][:, [j, j + len(TASKS)]].mean(axis=1)
            continue
        lr = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=seed)
        lr.fit(X["val"][m][:, [j, j + len(TASKS)]], y["val"][m, j])
        for r in probs:
            probs[r][:, j] = lr.predict_proba(X[r][:, [j, j + len(TASKS)]])[:, 1]
    return probs, fallback


def write_meta(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    os.replace(tmp, path)


def load_split_labels():
    return pd.read_parquet(CLEAN).set_index("smiles")[TASKS]
