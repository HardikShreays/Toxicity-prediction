"""B3/B4 - GNN training: masked loss, seeded training, early stopping, C3 prediction export.

CLI:  python -m tox.train_gnn --model mpnn --split scaffold_det --seed 0

Protocol: train on `train`, early-stop and select the best epoch on `val` (mean PR-AUC from the
shared `tox.metrics.score`), then restore the best weights and predict `val` and `test`. The test
set is only touched by `predict`, after training has finished; `fit` never receives it.
"""
from __future__ import annotations

import argparse
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from tox.constants import CLEAN, CONFIGS, PREDS, RESULTS, SPLITS, TASKS
from tox.featurize_graph import build_graphs
from tox.metrics import score
from tox.models_gnn import MODELS, build_model

GNN_RESULTS = RESULTS / "gnn"                       # run logs / curves; checkpoints (*.pt) are git-ignored


POS_WEIGHT_CAP = 30.0            # plan B3: pos_weight is clipped at 30 for stability


def masked_bce(logits: torch.Tensor, y: torch.Tensor,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-task binary cross-entropy that ignores missing labels.

    logits: [B, T] raw scores.  y: [B, T] with 1.0 / 0.0 / NaN (NaN = not measured).
    pos_weight: optional [T] weight on the positive term of each task (as in
    `F.binary_cross_entropy_with_logits`), computed from TRAINING labels only.

    Returns the mean loss over labelled entries only, so a batch with many missing labels is
    not diluted. Missing entries contribute exactly 0 to the loss and 0 to the gradient. A batch
    with no labels at all returns 0 (finite, with zero gradients) instead of NaN.

    Why not just call BCE on y: NaN targets would make the loss NaN. We replace NaN by a dummy 0
    *before* the BCE (so no NaN is ever computed) and then multiply by the mask, which zeroes
    both the value and the gradient of those entries.
    """
    if logits.shape != y.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and y {tuple(y.shape)} must match")
    mask = ~torch.isnan(y)
    y_safe = torch.where(mask, y, torch.zeros_like(y))
    per_entry = F.binary_cross_entropy_with_logits(
        logits, y_safe, pos_weight=pos_weight, reduction="none")
    per_entry = per_entry * mask                      # missing -> 0 value, 0 gradient
    n = mask.sum().clamp(min=1)                       # avoid 0/0 when nothing is labelled
    return per_entry.sum() / n


masked_bce_with_logits = masked_bce   # earlier name (B3 first version); kept for existing callers


def compute_pos_weight(y_train: np.ndarray, clip_max: float = POS_WEIGHT_CAP,
                       clip_min: float | None = None) -> torch.Tensor:
    """Plan B3: pos_weight[j] = n_neg / n_pos for task j, from TRAINING labels only, capped at 30.

    NaN labels are ignored in both counts. Only an upper cap is specified by the plan, so
    clip_min defaults to None (no lower bound). A task with no positive training labels gets
    weight 1.0 (neutral); the plan does not cover this case.
    """
    pos = np.nansum(y_train == 1, axis=0).astype(float)
    neg = np.nansum(y_train == 0, axis=0).astype(float)
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(w, clip_min, clip_max), dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_probs(model, graphs, batch_size: int = 256) -> np.ndarray:
    """Sigmoid probabilities [n, 12], in the order of `graphs`."""
    model.eval()
    out = [torch.sigmoid(model(b)).cpu().numpy() for b in DataLoader(graphs, batch_size=batch_size)]
    return np.concatenate(out).astype(np.float64)


def labels_of(graphs) -> np.ndarray:
    return torch.cat([g.y for g in graphs]).numpy()


def val_score(model, graphs) -> float:
    """Mean PR-AUC over the endpoints that can be scored (via the shared scorer)."""
    s = score(labels_of(graphs), predict_probs(model, graphs))
    return float(np.nanmean(s.pr_auc))


def fit(model, train_graphs, val_graphs, cfg: dict, seed: int, pos_weight=None) -> dict:
    """Train with early stopping on validation mean PR-AUC; restores the best weights.

    Only train and validation graphs are accepted here, so test labels cannot influence
    training, early stopping or checkpoint selection.
    """
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(train_graphs, batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), drop_last=len(train_graphs) > cfg["batch_size"])
    best, best_epoch, best_state, history = -np.inf, -1, None, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        losses = []
        for b in loader:
            opt.zero_grad()
            loss = masked_bce(model(b), b.y, pos_weight)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        v = val_score(model, val_graphs)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pr_auc": v})
        if v > best:
            best, best_epoch, best_state = v, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_pr_auc": best, "epochs_run": len(history), "history": history}


def load_split_graphs(split: str) -> dict:
    """role -> graphs for a C2 split, with labels from C1 (NaN kept for missing)."""
    sp = json.loads((SPLITS / f"{split}.json").read_text())
    labels = pd.read_parquet(CLEAN).set_index("smiles")[TASKS]
    return {r: build_graphs(sp[r], labels.loc[sp[r]].to_numpy(dtype=np.float32))
            for r in ("train", "val", "test")}


def export_predictions(model, graphs_by_role: dict, path) -> pd.DataFrame:
    """Write a contract-C3 file: smiles, role + 12 probability columns named exactly as TASKS."""
    frames = []
    for role in ("val", "test"):
        g = graphs_by_role[role]
        probs = predict_probs(model, g)
        frames.append(pd.DataFrame({"smiles": [x.smiles for x in g], "role": role,
                                    **{t: probs[:, j] for j, t in enumerate(TASKS)}}))
    df = pd.concat(frames, ignore_index=True)
    assert list(df.columns) == ["smiles", "role"] + TASKS
    assert df[TASKS].to_numpy().min() >= 0 and df[TASKS].to_numpy().max() <= 1
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_config(overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load((CONFIGS / "gnn.yaml").read_text())
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def run(model_name: str, split: str, seed: int, cfg: dict, out=PREDS, log_dir=GNN_RESULTS,
        graphs=None) -> dict:
    graphs = graphs or load_split_graphs(split)
    pw = None
    if cfg["pos_weight"]["enabled"]:
        hi = cfg["pos_weight"].get("clip_max")
        if hi is None:
            raise ValueError("pos_weight.enabled needs clip_max (the plan specifies 30)")
        pw = compute_pos_weight(labels_of(graphs["train"]), clip_max=hi)  # train labels only
    set_seed(seed)
    model = build_model(model_name, **{k: cfg[k] for k in
                        ("hidden", "n_layers", "pooling", "dropout", "use_bond_features")})
    info = fit(model, graphs["train"], graphs["val"], cfg, seed, pw)
    export_predictions(model, graphs, out / f"{model_name}__{split}__seed{seed}.parquet")
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}__{split}__seed{seed}"
    (log_dir / f"{stem}.json").write_text(json.dumps(
        {"model": model_name, "split": split, "seed": seed, "config": cfg,
         "pos_weight": None if pw is None else pw.tolist(), **info}, indent=1))
    torch.save(model.state_dict(), log_dir / f"{stem}.pt")               # git-ignored (*.pt)
    print(f"{stem}: best epoch {info['best_epoch']}, val mean PR-AUC {info['best_val_pr_auc']:.3f}",
          flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, help="override max_epochs (smoke tests)")
    args = ap.parse_args()
    run(args.model, args.split, args.seed, load_config({"max_epochs": args.epochs}))


if __name__ == "__main__":
    main()
