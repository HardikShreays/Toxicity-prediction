"""Technical validation (plan, Phase 2 rubric "Technical validation", Hardik): paired
significance tests and calibration checks on the saved C4 / C3 files.

Wilcoxon signed-rank: for one split family (scaffold, random or scaffold_det) and metric,
each endpoint contributes its mean over that family's runs, giving 12 paired observations
per model pair. The test asks whether one model's per-endpoint advantage is systematic or
noise; with 12 endpoints it is exactly the n the plan intends. P-values are exploratory
(no multiplicity correction - we say so in the report).

Calibration: reliability curves per model on the test predictions, pooled over the labelled
(task, molecule) pairs of its runs. Reported per model: ECE (expected calibration error,
10 equal-width bins) and Brier score, plus one reliability diagram per model. Written to
results/validation/ and results/figures/.

  python -m tox.validate --wilcoxon --calibration
  python -m tox.validate --wilcoxon --families scaffold random
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as scipy_wilcoxon
from sklearn.metrics import brier_score_loss

from tox.aggregate import load_all
from tox.constants import CLEAN, FIGURES, PREDS, TASKS, VALIDATION
from tox.metrics import load_preds

FAMILIES = ("scaffold_det", "scaffold", "random")


def per_endpoint(df: pd.DataFrame, family: str, metric: str = "roc_auc") -> pd.DataFrame:
    """model x task: mean of `metric` over the runs of one split family (12 paired columns)."""
    d = df[df.split == family]
    return d.pivot_table(index="model", columns="task", values=metric, aggfunc="mean")


def wilcoxon_pair(df: pd.DataFrame, a: str, b: str, family: str, metric: str = "roc_auc") -> dict:
    """Paired Wilcoxon signed-rank between two models on one family and metric.

    Returns per-model endpoint means, the signed differences, the statistic and the two-sided
    p-value (zero differences dropped with the default 'wilcox' handling). Requires both
    models to be scored on the same family (asserted).
    """
    tab = per_endpoint(df, family, metric)
    for m in (a, b):
        if m not in tab.index:
            raise KeyError(f"{m} has no {family} rows in the C4 files")
    x, y = tab.loc[a].to_numpy(dtype=float), tab.loc[b].to_numpy(dtype=float)
    stat, p = scipy_wilcoxon(x, y, alternative="two-sided")
    return {"model_a": a, "model_b": b, "family": family, "metric": metric,
            "n_endpoints": len(x), f"{metric}_a": float(np.nanmean(x)),
            f"{metric}_b": float(np.nanmean(y)), "statistic": float(stat), "p_value": float(p)}


def wilcoxon_table(df: pd.DataFrame, models: list[str] | None = None,
                   families=FAMILIES, metrics=("roc_auc", "pr_auc")) -> pd.DataFrame:
    """All model pairs x families x metrics -> results/validation/wilcoxon.csv."""
    models = models or sorted(set(df.model))
    rows = []
    for family in families:
        for a, b in itertools.combinations(models, 2):
            for metric in metrics:
                try:
                    rows.append(wilcoxon_pair(df, a, b, family, metric))
                except KeyError:
                    continue                      # a model without this family/metric: skip the pair
    return pd.DataFrame(rows)


def reliability(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """One reliability curve from pooled labelled (task, molecule) pairs: bin edges, the mean
    predicted probability and the observed positive fraction per bin, and ECE."""
    m = ~np.isnan(y_true)
    y, p = y_true[m], y_prob[m]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        k = idx == b
        if not k.any():
            continue
        conf, frac = float(p[k].mean()), float(y[k].mean())
        ece += k.mean() * abs(frac - conf)
        rows.append({"bin": b, "bin_low": edges[b], "bin_high": edges[b + 1], "count": int(k.sum()),
                     "mean_predicted": conf, "fraction_positive": frac})
    return pd.DataFrame(rows), float(ece)


def calibration_table(models: list[str] | None = None, n_bins: int = 10,
                      preds_dir=None, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """ECE and Brier score per model, pooled over its test runs' labelled pairs."""
    preds_dir = PREDS if preds_dir is None else preds_dir
    labels = (pd.read_parquet(CLEAN).set_index("smiles")[TASKS] if labels is None else labels)
    rows = []
    for model in sorted({f.stem.split("__")[0] for f in preds_dir.glob("*__*__seed*.parquet")}):
        if models is not None and model not in models:
            continue
        ys, ps = [], []
        for f in sorted(preds_dir.glob(f"{model}__*__seed*.parquet")):
            y, p = load_preds(f, "test", labels)
            ys.append(y.reshape(-1))
            ps.append(p.reshape(-1))
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        keep = ~np.isnan(y)
        _rel, ece = reliability(y[keep], p[keep], n_bins)
        rows.append({"model": model, "n_pairs": int(keep.sum()), "ece": ece,
                     "brier": float(brier_score_loss(y[keep], p[keep]))})
    return pd.DataFrame(rows)


def plot_calibration(models: list[str], n_bins: int = 10) -> None:
    """One reliability diagram PNG per model: observed positive rate vs predicted probability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = pd.read_parquet(CLEAN).set_index("smiles")[TASKS]
    for model in models:
        ys, ps = [], []
        for f in sorted(PREDS.glob(f"{model}__*__seed*.parquet")):
            y, p = load_preds(f, "test", labels)
            ys.append(y.reshape(-1))
            ps.append(p.reshape(-1))
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        keep = ~np.isnan(y)
        rel, _ = reliability(y[keep], p[keep], n_bins)
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
        ax.plot(rel.mean_predicted, rel.fraction_positive, marker="o", ms=3)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted probability",
               ylabel="observed positive fraction",
               title=f"Reliability: {model}")
        fig.tight_layout()
        out = FIGURES / f"calibration_{model}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Wilcoxon tests + calibration curves (Hardik).")
    ap.add_argument("--wilcoxon", action="store_true")
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    if args.wilcoxon:
        df = load_all()
        tab = wilcoxon_table(df, args.models, args.families)
        VALIDATION.mkdir(parents=True, exist_ok=True)
        tab.to_csv(VALIDATION / "wilcoxon.csv", index=False)
        sig = tab[tab.p_value < 0.05].sort_values("p_value")
        print(f"{len(tab)} pairs tested; {len(sig)} significant at 0.05 -> "
              f"{VALIDATION / 'wilcoxon.csv'}")
    if args.calibration:
        models = args.models or sorted(
            {f.stem.split("__")[0] for f in PREDS.glob("*__*__seed*.parquet")})
        tab = calibration_table(models)
        VALIDATION.mkdir(parents=True, exist_ok=True)
        tab.to_csv(VALIDATION / "calibration.csv", index=False)
        plot_calibration(models)
        print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
