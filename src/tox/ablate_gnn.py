"""B5 - GNN architecture ablations (plan p. 15-16): all on scaffold_det, 3 seeds, one thing changed.

    python -m tox.ablate_gnn --model mpnn                 # all applicable ablations
    python -m tox.ablate_gnn --model mpnn --groups layers bond
    python -m tox.metrics --preds results/ablations/gnn/preds --out results/ablations/gnn/metrics --role test
    python -m tox.ablate_gnn --model mpnn --summary       # -> summary.csv and per_task.csv

The five ablations and the file tokens they write (baseline = the main run `{model}` from
`python -m tox.train_gnn --grid`, so a pair is e.g. mpnn vs mpnn-noedge as in the plan):
    layers    {model}-l2 {model}-l3 {model}-l4 {model}-l5   (all four are run; none is assumed baseline)
    bond      {model}-noedge       bond features zeroed (mpnn, afp only; GCN never sees bonds)
    pooling   {model}-pmean {model}-pmax      (baseline readout is mean||max; gcn, mpnn only)
    posweight {model}-nopw         pos_weight off (baseline: on, capped at 30)
    single    {model}-single       12 separate single-output models, one per endpoint, instead of
                                   one 12-output model (does label sharing help sparse endpoints?)

Everything else comes from the model's FROZEN tuned config (results/gnn_params.json), so
`tune_gnn` must have run first. Files go to results/ablations/gnn/{preds,logs,figures}: a
sub-folder, because tox.ablate_fp's summary reads every CSV in results/ablations/metrics/.
Scoring is done only by `tox.metrics`. Test labels are never used for any choice.

ASSUMPTIONS (the plan does not say): the architecture the ablations run on is a parameter
(default mpnn, the only one the plan names, for the bond ablation); the single-task models use
seed*100+k for endpoint k and early-stop on that endpoint's validation ROC-AUC.
"""
from __future__ import annotations

import argparse
import copy
import json

import re

import numpy as np
import pandas as pd
import torch

from tox.constants import ABLATIONS, METRICS, SEEDS, TASKS
from tox.models_gnn import build_model, canonical_name
from tox.train_gnn import (fit, frozen_config, load_split_graphs, make_pos_weight, predict_probs,
                           run, set_seed, write_c3)

OUT = ABLATIONS / "gnn"
SPLIT = "scaffold_det"
GROUPS = ("layers", "bond", "pooling", "posweight", "single")
ARCH_KEYS = ("hidden", "n_layers", "pooling", "dropout", "use_bond_features")


def variants(model: str, groups=None) -> dict:
    """tag -> config overrides, for the requested groups (default: every group that applies)."""
    model = canonical_name(model)
    applicable = {"layers": True, "bond": model in ("mpnn", "afp"),
                  "pooling": model in ("gcn", "mpnn"), "posweight": True, "single": True}
    if groups is None:
        groups = [g for g in GROUPS if applicable[g]]
    out = {}
    for g in groups:
        if g not in GROUPS:
            raise ValueError(f"unknown ablation group {g!r}; choose from {GROUPS}")
        if not applicable[g]:
            raise ValueError(f"ablation {g!r} does not apply to {model!r} "
                             "(gcn ignores bond features; afp has its own readout)")
        if g == "layers":
            out.update({f"{model}-l{n}": {"n_layers": n} for n in (2, 3, 4, 5)})
        elif g == "bond":
            out[f"{model}-noedge"] = {"use_bond_features": False}
        elif g == "pooling":
            out.update({f"{model}-pmean": {"pooling": "mean"}, f"{model}-pmax": {"pooling": "max"}})
        elif g == "posweight":
            out[f"{model}-nopw"] = {"pos_weight": {"enabled": False, "clip_max": 30}}
        else:
            out[f"{model}-single"] = {"single_task": True}
    return out


def run_single_task(model_name: str, split: str, seed: int, cfg: dict, out, log_dir, graphs,
                    fig_dir=None, tag: str | None = None) -> dict:
    """12 independent single-output models (one per endpoint), merged into one C3 file.

    Same architecture, config, loss and early stopping as the multi-task run, but each model sees
    only its own endpoint (labels of the others are unused). pos_weight, if on, is that
    endpoint's train-only weight.
    """
    model_name = canonical_name(model_name)
    stem = f"{tag or model_name}__{split}__seed{seed}"
    pw = make_pos_weight(cfg, graphs["train"])
    probs = {r: np.zeros((len(graphs[r]), len(TASKS))) for r in ("val", "test")}
    per_task = []
    for k, t in enumerate(TASKS):
        task_seed = seed * 100 + k
        set_seed(task_seed)
        model = build_model(model_name, n_tasks=1, **{a: cfg[a] for a in ARCH_KEYS})
        info = fit(model, graphs["train"], graphs["val"], cfg, task_seed, pw,
                   ckpt_dir=log_dir / "ckpt" / stem / f"task{k}", task=k)
        for r in ("val", "test"):              # test is predicted only after this endpoint's training ended
            probs[r][:, k] = predict_probs(model, graphs[r])[:, 0]
        per_task.append({"task": t, "seed": task_seed, "best_epoch": info["best_epoch"],
                         "epochs_run": info["epochs_run"], "val_roc_auc": info["best_val_roc_auc"],
                         "val_pr_auc": info["best_val_pr_auc"]})
    write_c3(graphs, probs, out / f"{stem}.parquet")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{stem}.json").write_text(json.dumps(
        {"model": model_name, "tag": tag or model_name, "split": split, "seed": seed, "config": cfg,
         "single_task": True, "versions": {"torch": torch.__version__},
         "pos_weight": None if pw is None else pw.tolist(), "tasks": per_task}, indent=1))
    print(f"{stem}: 12 single-task models, mean val ROC-AUC "
          f"{np.nanmean([p['val_roc_auc'] for p in per_task]):.3f}", flush=True)
    return {"tasks": per_task}


def run_ablations(model: str, groups=None, seeds=SEEDS, force: bool = False, out_dir=None,
                  params_path=None, graphs=None, runner=None, single_runner=None,
                  max_epochs: int | None = None) -> list:
    """Run every variant for every seed on scaffold_det; resumable (skips existing C3 files)."""
    model = canonical_name(model)
    out_dir = OUT if out_dir is None else out_dir
    runner, single_runner = runner or run, single_runner or run_single_task
    base = frozen_config(model, params_path)
    if max_epochs is not None:
        base["max_epochs"] = max_epochs
    todo = variants(model, groups)
    graphs = graphs or load_split_graphs(SPLIT)
    done = []
    for tag, overrides in todo.items():
        for seed in seeds:
            if not force and (out_dir / "preds" / f"{tag}__{SPLIT}__seed{seed}.parquet").exists():
                continue
            cfg = copy.deepcopy(base)
            cfg.update(copy.deepcopy(overrides))
            kwargs = dict(out=out_dir / "preds", log_dir=out_dir / "logs", graphs=graphs,
                          fig_dir=out_dir / "figures", tag=tag)
            if cfg.pop("single_task", False):
                single_runner(model, SPLIT, seed, cfg, **kwargs)
            else:
                runner(model, SPLIT, seed, cfg, **kwargs)
            done.append((tag, seed))
    return done


SUFFIX_GROUPS = [(r"-l\d+$", "layers"), (r"-noedge$", "bond"), (r"-p(mean|max)$", "pooling"),
                 (r"-nopw$", "posweight"), (r"-single$", "single")]


def group_of(tag: str, model: str) -> str:
    """Ablation group of a file token; the main run `{model}` itself is the 'baseline' arm."""
    if tag == model:
        return "baseline"
    for pattern, group in SUFFIX_GROUPS:
        if re.search(pattern, tag):
            return group
    raise ValueError(f"{tag!r} is not a {model} ablation token")


def _read_c4(model: str, metrics_dir=None, baseline_dir=None) -> pd.DataFrame:
    """Scored (C4) files for the ablation variants plus the baseline arm, scaffold_det only.
    Baseline = the main `{model}` runs already scored in results/metrics (written by tox.metrics)."""
    model = canonical_name(model)
    metrics_dir = OUT / "metrics" if metrics_dir is None else metrics_dir
    baseline_dir = METRICS if baseline_dir is None else baseline_dir
    files = sorted(metrics_dir.glob(f"{model}-*__{SPLIT}__seed*.csv")) \
        + sorted(baseline_dir.glob(f"{model}__{SPLIT}__seed*.csv"))
    if not files:
        raise FileNotFoundError(f"no scored {model} ablation files in {metrics_dir}; run tox.metrics first")
    frames = []
    for f in files:
        tag, split, seed = f.stem.split("__")
        frames.append(pd.read_csv(f).assign(variant=tag, group=group_of(tag, model), seed=int(seed[4:])))
    return pd.concat(frames, ignore_index=True)


def summarize(model: str, metrics_dir=None, baseline_dir=None) -> pd.DataFrame:
    """One row per variant: 12-task mean per seed, then mean/std over seeds (the headline.csv recipe,
    so numbers are comparable to the main table). n_runs shows how many seeds were available."""
    df = _read_c4(model, metrics_dir, baseline_dir)
    per_run = df.groupby(["group", "variant", "seed"], sort=False)[["roc_auc", "pr_auc", "pos_rate"]].mean()
    out = (per_run.groupby(["group", "variant"], sort=False)
                  .agg(n_runs=("roc_auc", "size"),
                       roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                       pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                       pos_rate=("pos_rate", "mean"))
                  .reset_index())
    order = {g: i for i, g in enumerate(["baseline", *GROUPS])}
    return out.sort_values(["group", "variant"], key=lambda c: c.map(order) if c.name == "group" else c,
                           ignore_index=True)


def per_task_summary(model: str, metrics_dir=None, baseline_dir=None) -> pd.DataFrame:
    """Per endpoint and variant: seed-mean ROC/PR-AUC plus the endpoint's positive rate and count
    (to see whether label sharing helps the sparse endpoints, single vs baseline multi-task)."""
    df = _read_c4(model, metrics_dir, baseline_dir)
    return (df.groupby(["group", "variant", "task"], sort=False)
              .agg(n_runs=("seed", "nunique"), n_pos=("n_pos", "mean"), pos_rate=("pos_rate", "mean"),
                   roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                   pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"))
              .reset_index())


def main():
    ap = argparse.ArgumentParser(description="GNN ablations on scaffold_det (plan B5).")
    ap.add_argument("--model", default="mpnn", help="architecture to ablate (default mpnn: ASSUMPTION)")
    ap.add_argument("--groups", nargs="+", choices=GROUPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the variants and exit")
    ap.add_argument("--summary", action="store_true", help="roll the scored C4 files up into summary.csv / per_task.csv")
    ap.add_argument("--epochs", type=int, help="override max_epochs (smoke tests only)")
    a = ap.parse_args()
    if a.list:
        for tag, ov in variants(a.model, a.groups).items():
            print(f"{tag:22s} {ov}")
        return
    if a.summary:
        summary = summarize(a.model)
        summary.to_csv(OUT / "summary.csv", index=False)
        per_task_summary(a.model).to_csv(OUT / "per_task.csv", index=False)
        print(summary.round(3).to_string(index=False))
        return
    run_ablations(a.model, a.groups, a.seeds, a.force, max_epochs=a.epochs)


if __name__ == "__main__":
    main()
