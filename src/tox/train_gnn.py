"""B3/B4 - GNN training: masked loss, seeded training, early stopping, C3 prediction export.

CLI:  python -m tox.train_gnn --model mpnn --split scaffold_det --seed 0

Protocol (plan B4): train on `train`; early-stop on `val` mean ROC-AUC (patience 25, max 200
epochs), computed with the shared `tox.metrics.score`; restore the best-epoch weights; then
predict `val` and `test`. Validation mean PR-AUC is logged every epoch too, but only as the
metric for choosing between hyper-parameter configs, never for stopping. The test set is only
touched by `export_predictions`, after training has finished; `fit` never receives it.

Checkpoints: with `ckpt_dir` set, `fit` writes `last.pt` after EVERY epoch (model, optimizer, RNG
states, history, early-stopping counters) and `best.pt` whenever validation ROC-AUC improves, so
a disconnected session (Colab) resumes exactly where it stopped. *.pt files are git-ignored.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from tox.constants import CLEAN, CONFIGS, FIGURES, PREDS, RESULTS, SPLITS, TASKS
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


def val_scores(model, graphs) -> dict:
    """Validation mean ROC-AUC and mean PR-AUC over the endpoints that can be scored.

    Both come from the shared scorer. ROC-AUC drives early stopping; PR-AUC is what we use to
    choose between hyper-parameter configs (plan B4 / H3.4).
    """
    s = score(labels_of(graphs), predict_probs(model, graphs))
    return {"roc_auc": float(np.nanmean(s.roc_auc)), "pr_auc": float(np.nanmean(s.pr_auc))}


def _atomic_save(obj, path) -> None:
    """Write via a temp file so a disconnect mid-write cannot corrupt the checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])


def fit(model, train_graphs, val_graphs, cfg: dict, seed: int, pos_weight=None,
        ckpt_dir=None) -> dict:
    """Train with early stopping on validation mean ROC-AUC; restores the best-epoch weights.

    Stops when `patience` epochs pass without a new best, or after `max_epochs`. Only train and
    validation graphs are accepted here, so test labels cannot influence training, early
    stopping or checkpoint selection. If `ckpt_dir` is given the run is checkpointed every epoch
    and resumes from `last.pt` when it exists (same cfg and seed required).
    """
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_epoch, best_state, history, start, done = -np.inf, -1, None, [], 0, False
    last_path = None if ckpt_dir is None else ckpt_dir / "last.pt"
    if last_path is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if last_path.exists():
            # our own file (contains numpy RNG state), so full unpickling is intended
            ck = torch.load(last_path, weights_only=False)
            if ck["seed"] != seed or ck["cfg"] != cfg:
                raise ValueError(f"{last_path} was written with a different seed/config; "
                                 "delete it or use another log_dir")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            _set_rng_state(ck["rng"])
            best, best_epoch, best_state = ck["best"], ck["best_epoch"], ck["best_state"]
            history, start, done = ck["history"], ck["epoch"] + 1, ck["done"]

    for epoch in range(start, 0 if done else cfg["max_epochs"]):
        model.train()
        # shuffling depends only on (seed, epoch), so a resumed run sees the same batches
        loader = DataLoader(train_graphs, batch_size=cfg["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(seed * 100003 + epoch),
                            drop_last=len(train_graphs) > cfg["batch_size"])
        losses = []
        for b in loader:
            opt.zero_grad()
            loss = masked_bce(model(b), b.y, pos_weight)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        v = val_scores(model, val_graphs)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_roc_auc": v["roc_auc"], "val_pr_auc": v["pr_auc"]})
        roc = v["roc_auc"] if np.isfinite(v["roc_auc"]) else -np.inf   # unscorable val never "improves"
        improved = roc > best
        if improved:
            best, best_epoch, best_state = roc, epoch, copy.deepcopy(model.state_dict())
        stop = (not improved) and best_epoch >= 0 and epoch - best_epoch >= cfg["patience"]
        if last_path is not None:
            if improved:
                _atomic_save({"model": best_state, "epoch": epoch, "val_roc_auc": roc}, ckpt_dir / "best.pt")
            _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": _rng_state(),
                          "best": best, "best_epoch": best_epoch, "best_state": best_state,
                          "history": history, "epoch": epoch, "seed": seed, "cfg": cfg,
                          "done": stop or epoch + 1 >= cfg["max_epochs"]}, last_path)
        if stop:
            break
    if best_state is None:
        raise RuntimeError("validation ROC-AUC was never computable; check the val split labels")
    model.load_state_dict(best_state)
    h = history[best_epoch]
    return {"best_epoch": best_epoch, "best_val_roc_auc": h["val_roc_auc"],
            "best_val_pr_auc": h["val_pr_auc"],           # config-selection metric, at the ROC-best epoch
            "epochs_run": len(history), "history": history}


def plot_curves(info: dict, path) -> None:
    """Learning curves (train loss, val ROC-AUC) as a PNG; the best epoch is marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in info["history"]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ep, [h["train_loss"] for h in info["history"]])
    ax[0].set(xlabel="epoch", ylabel="train masked BCE", title="Training loss")
    ax[1].plot(ep, [h["val_roc_auc"] for h in info["history"]])
    ax[1].axvline(info["best_epoch"], ls="--", c="gray", label=f"best epoch {info['best_epoch']}")
    ax[1].set(xlabel="epoch", ylabel="val mean ROC-AUC", title="Validation")
    ax[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


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
        graphs=None, fig_dir=None) -> dict:
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
    stem = f"{model_name}__{split}__seed{seed}"
    info = fit(model, graphs["train"], graphs["val"], cfg, seed, pw, ckpt_dir=log_dir / "ckpt" / stem)
    export_predictions(model, graphs, out / f"{stem}.parquet")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{stem}.json").write_text(json.dumps(
        {"model": model_name, "split": split, "seed": seed, "config": cfg,
         "early_stopping": {"metric": "val_mean_roc_auc", "patience": cfg["patience"],
                            "max_epochs": cfg["max_epochs"]},
         "versions": {"torch": torch.__version__},
         "pos_weight": None if pw is None else pw.tolist(), **info}, indent=1))
    torch.save(model.state_dict(), log_dir / f"{stem}.pt")               # best weights; git-ignored
    plot_curves(info, (FIGURES if fig_dir is None else fig_dir) / f"{stem}_curve.png")
    print(f"{stem}: best epoch {info['best_epoch']}, val mean ROC-AUC {info['best_val_roc_auc']:.3f}, "
          f"PR-AUC {info['best_val_pr_auc']:.3f}", flush=True)
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
