"""B4 - hyper-parameter tuning on the scaffold_det VALIDATION set only.

Grid (plan B4): hidden in {64,128,256} x layers in {3,4} x dropout in {0.1,0.3} = 12 configs per
model. Each config is trained with early stopping on val mean ROC-AUC (same as the final runs);
configs are COMPARED on val mean PR-AUC (plan B4 + H3.4). The winner is frozen in
results/gnn_params.json, then `python -m tox.train_gnn --grid` runs it on every split and seed.

Test labels are never loaded here. Assumption (the plan does not say): one seed (0) per config.
Resumable: finished trials are stored after each config and skipped on a re-run.

CLI:  python -m tox.tune_gnn [--models gcn mpnn afp]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import torch

from tox.models_gnn import MODELS, build_model, canonical_name
from tox.train_gnn import (GNN_PARAMS, TUNED_KEYS, fit, load_config, load_split_graphs,
                           make_pos_weight, set_seed)

GRID = {"hidden": [64, 128, 256], "n_layers": [3, 4], "dropout": [0.1, 0.3]}
SPLIT = "scaffold_det"
SELECTION_METRIC = "val_mean_pr_auc"


def _save(state: dict, path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, path)


def tune(model_name: str, graphs=None, grid=None, seed: int = 0, base_cfg=None, params_path=None,
         log=print) -> dict:
    """Run the grid for one model; return and persist the best params (highest val PR-AUC)."""
    model_name = canonical_name(model_name)
    grid = GRID if grid is None else grid
    path = GNN_PARAMS if params_path is None else params_path
    base_cfg = base_cfg or load_config()
    graphs = graphs or load_split_graphs(SPLIT, roles=("train", "val"))   # test is never loaded
    pw = make_pos_weight(base_cfg, graphs["train"])         # train labels only

    state = json.loads(path.read_text()) if path.exists() else {}
    entry = state.setdefault(model_name, {})
    entry.update({"split": SPLIT, "seed": seed, "selection_metric": SELECTION_METRIC,
                  "early_stopping": "val_mean_roc_auc", "grid": grid})
    trials = entry.setdefault("trials", [])
    seen = {tuple(t[k] for k in TUNED_KEYS) for t in trials}

    for values in itertools.product(*(grid[k] for k in TUNED_KEYS)):
        if values in seen:
            continue
        params = dict(zip(TUNED_KEYS, values))
        cfg = dict(base_cfg, **params)
        set_seed(seed)
        model = build_model(model_name, **{k: cfg[k] for k in
                            ("hidden", "n_layers", "pooling", "dropout", "use_bond_features")})
        info = fit(model, graphs["train"], graphs["val"], cfg, seed, pw)    # no test input exists
        trials.append({**params, "val_pr_auc": info["best_val_pr_auc"],
                       "val_roc_auc": info["best_val_roc_auc"], "best_epoch": info["best_epoch"],
                       "epochs_run": info["epochs_run"], "torch": torch.__version__})
        best_trial = max(trials, key=lambda t: t["val_pr_auc"])        # first max wins ties
        entry["best"] = {k: best_trial[k] for k in TUNED_KEYS} | {"val_pr_auc": best_trial["val_pr_auc"]}
        _save(state, path)                                             # resumable after every config
        log(f"{model_name} {params}: val PR-AUC {info['best_val_pr_auc']:.3f} "
            f"(ROC {info['best_val_roc_auc']:.3f}, {info['epochs_run']} epochs)")
    return entry["best"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=sorted(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, help="override max_epochs (smoke tests only)")
    args = ap.parse_args()
    graphs = load_split_graphs(SPLIT, roles=("train", "val"))
    for m in args.models:
        print(m, "best:", tune(m, graphs, seed=args.seed, base_cfg=load_config({"max_epochs": args.epochs})))


if __name__ == "__main__":
    main()
