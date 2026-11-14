"""Phase 2 hybrid ablations that are analyses of saved predictions (plan p. 19, Hardik's
share of the ablation grid together with tox.train_hybrid's training arms):

  sparse_dense  hybrid vs parents on the 4 sparsest vs 4 densest endpoints - tests the
                stated rationale ("the hybrid should help most on the sparse endpoints")
                directly, from the C4 files the shared scorer wrote.
  sim_bins      per test molecule, the maximum Tanimoto similarity (Morgan 2048, the same
                fingerprint everywhere) to any training molecule; binned into
                [0, 0.3), [0.3, 0.5), [0.5, 1.0]; ROC-AUC per bin per model pooled over
                tasks with the mask - tests whether the hybrid's gain comes from the hard,
                dissimilar molecules.

Nothing here computes a new model or metric type: both read the shared scorer's C4 output
(tox.metrics) and label tensors from C1, aggregate with the usual recipes, and write
results/hybrid/{sparse_dense.csv, sim_bins.csv} plus figures.

  python -m tox.hybrid_analysis             # both analyses for every scored model
  python -m tox.hybrid_analysis --split random_s0
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import roc_auc_score

from tox.constants import CLEAN, FIGURES, HYBRID, METRICS, PREDS, TASKS
from tox.featurize_fp import load_features
from tox.metrics import load_preds
from tox.splits import load

PARENTS = ["lr_fp", "lr_fpdesc", "rf_fp", "rf_fpdesc", "xgb_fp", "xgb_fpdesc",
           "gcn", "mpnn", "afp", "hyb_fusion", "hyb_embxgb", "hyb_stack"]
BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 1.0 + 1e-9)]     # plan P2: [0,0.3), [0.3,0.5), [0.5,1.0]
SPLIT = "scaffold_det"


def _c4_frame(models: list[str], split: str = SPLIT, metrics_dir=None) -> pd.DataFrame:
    """All scored runs of `models` on one split, from the C4 files (no new metrics)."""
    metrics_dir = METRICS if metrics_dir is None else metrics_dir
    frames = []
    for f in sorted(metrics_dir.glob("*__" + split + "__seed*.csv")):
        if f.stem.split("__")[0] not in models:
            continue
        tag, _s, seed = f.stem.split("__")
        frames.append(pd.read_csv(f).assign(model=tag, seed=int(seed[4:])))
    if not frames:
        raise FileNotFoundError(f"no scored C4 files for {models} on {split} in {metrics_dir}; "
                                "run the grids and tox.metrics / tox.train_hybrid --score first")
    return pd.concat(frames, ignore_index=True)


def sparse_dense(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Mean ROC-AUC / PR-AUC per model over the k sparsest vs k densest endpoints.

    Sparsity is the endpoint's mean n_pos across the runs in `df` (train+val+test positive
    count as recorded by the scorer for the test role - the honest size of the labelled
    positive class a model is tested on). One row per (model, group): mean over endpoints and
    seeds, so the two groups are directly comparable.
    """
    n_pos = df.groupby("task").n_pos.mean()
    order = n_pos.sort_values()
    sparse, dense = list(order.index[:k]), list(order.index[-k:])
    out = (df[df.task.isin(sparse + dense)]
           .assign(group=lambda d: np.where(d.task.isin(sparse), f"{k} sparsest",
                                            f"{k} densest"))
           .groupby(["model", "group"], sort=False)
           .agg(n_runs=("seed", "nunique"), n_pos=("n_pos", "mean"),
                roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"))
           .reset_index())
    return out


def max_tanimoto(split: str, fp: np.ndarray) -> np.ndarray:
    """For every test molecule of `split`: max Tanimoto similarity to any train molecule.

    Uses the same Morgan fingerprint settings as the rest of the repo (radius and size from
    configs/aml.yaml, rdkit bit vectors for the Tanimoto computation). Bulk computation over
    the split's train set; returns one maximum per test molecule, in the split's test order.
    """
    from tox.featurize_fp import gen as morgan_gen

    s = load(split)
    df = pd.read_parquet(CLEAN).set_index("smiles")
    bv = {smi: morgan_gen.GetFingerprint(MolFromSmiles(smi)) for smi in df.index}
    train = [bv[smi] for smi in s["train"]]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(bv[smi], train)) for smi in s["test"]])


def sim_bins(split: str = SPLIT, models: list[str] | None = None, preds_dir=None) -> pd.DataFrame:
    """ROC-AUC per similarity bin per model on one split, pooled over tasks with the mask.

    Per-task numbers are too sparse in the low bins (the plan says so), so all labelled
    (task, molecule) pairs of the test role are pooled into one ROC-AUC per bin. The pooled
    value is a diagnostic for this analysis only: every per-task number still comes from the
    shared scorer's C4 files, as everywhere in the repo. One row per (model, bin): mean and
    std over the split's seeds.
    """
    models = models or PARENTS
    fp, _ = load_features()
    sims = max_tanimoto(split, fp)
    labels = pd.read_parquet(CLEAN).set_index("smiles")[TASKS]
    preds_dir = PREDS if preds_dir is None else preds_dir
    rows = []
    for f in sorted(preds_dir.glob("*__" + split + "__seed*.parquet")):
        tag = f.stem.split("__")[0]
        if tag not in models:
            continue
        y, probs = load_preds(f, "test", labels)
        for lo, hi in BINS:
            m = (sims >= lo) & (sims < hi)                    # test molecules in this bin
            yy = np.concatenate([y[m][:, j] for j in range(len(TASKS))])
            pp = np.concatenate([probs[m][:, j] for j in range(len(TASKS))])
            keep = ~np.isnan(yy)                              # missing labels excluded, as always
            rows.append({"model": tag, "seed": int(f.stem.split("__seed")[1]),
                         "bin": f"[{lo:g}, {hi:g})", "n_mols": int(m.sum()),
                         "roc_auc": float(roc_auc_score(yy[keep], pp[keep]))})
    if not rows:
        raise FileNotFoundError(f"no C3 prediction files for {models} on {split} in {preds_dir}")
    df = pd.DataFrame(rows)
    return (df.groupby(["model", "bin"], sort=False)
              .agg(n_mols=("n_mols", "first"), n_runs=("seed", "nunique"),
                   roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"))
              .reset_index())


def plot_sim_bins(sim: pd.DataFrame, path) -> None:
    """Bar chart: ROC-AUC by Tanimoto bin, fingerprints vs GNN vs hybrid (the plan P2 view)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [f"[{lo:g}, {hi:g})" for lo, hi in BINS]
    show = [m for m in ("rf_fpdesc", "xgb_fpdesc", "mpnn", "hyb_fusion", "hyb_embxgb", "hyb_stack")
            if m in set(sim.model)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(show)
    for i, m in enumerate(show):
        g = sim[sim.model == m].set_index("bin").loc[order]
        ax.bar(np.arange(len(order)) + i * w, g.roc_auc_mean, w,
               yerr=g.roc_auc_std.fillna(0), capsize=2, label=m)
    ax.set(xticks=np.arange(len(order)) + 0.4 - w / 2, xticklabels=order,
           xlabel="max Tanimoto similarity to the training set",
           ylabel="test ROC-AUC (pooled tasks)", title="Performance by similarity to train")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Hybrid ablation analyses (plan p. 19).")
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--k", type=int, default=4, help="endpoints per sparse/dense group")
    ap.add_argument("--models", nargs="+", default=PARENTS)
    args = ap.parse_args()

    df = _c4_frame(args.models, args.split)
    sd = sparse_dense(df, args.k)
    sim = sim_bins(args.split, args.models)
    HYBRID.mkdir(parents=True, exist_ok=True)
    sd.to_csv(HYBRID / "sparse_dense.csv", index=False)
    sim.to_csv(HYBRID / "sim_bins.csv", index=False)
    plot_sim_bins(sim, FIGURES / "hybrid_sim_bins.png")
    print(sd.round(3).to_string(index=False))
    print(sim.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
