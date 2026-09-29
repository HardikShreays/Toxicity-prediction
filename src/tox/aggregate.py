"""H3 - roll every C4 metrics file up into results/summary.csv and results/headline.csv.

split families: scaffold_det (3 model seeds), scaffold (scaffold_s0-2), random (random_s0-2).
summary  : model x split x task, mean/std over the 3 runs
headline : model x split, 12-task mean per run, then mean/std over the 3 runs
"""
import re

import pandas as pd

from tox.constants import HEADLINE, METRICS, SUMMARY


def load_all():
    frames = []
    for f in sorted(METRICS.glob("*.csv")):
        model, split, seed = f.stem.split("__")
        d = pd.read_csv(f)
        d["model"], d["split"], d["seed"] = model, re.sub(r"_s\d+$", "", split), int(seed[4:])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()
    summary = (df.groupby(["model", "split", "task"], sort=False)
                 .agg(n_runs=("roc_auc", "size"), pos_rate=("pos_rate", "mean"),
                      roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                      pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"))
                 .reset_index())
    summary.round(4).to_csv(SUMMARY, index=False)

    per_run = df.groupby(["model", "split", "seed"], sort=False)[["roc_auc", "pr_auc", "pos_rate"]].mean()
    headline = (per_run.groupby(["model", "split"], sort=False)
                       .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                            pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                            pos_rate=("pos_rate", "mean"))
                       .reset_index())
    headline.round(4).to_csv(HEADLINE, index=False)
    print(headline.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
