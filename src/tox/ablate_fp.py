"""A3 (feature-engineering rubric line) - fingerprint ablation with XGBoost on scaffold_det, 3 seeds:
radius 1/2/3 x 1024/2048 bits, plus descriptors only. Everything else is the same as xgb_fp.
Preds -> results/ablations/preds, then score with
python -m tox.metrics --preds results/ablations/preds --out results/ablations/metrics
and roll the scores up into results/ablations/summary.csv (the report's ablation table) with
python -m tox.ablate_fp --summary
"""
import argparse
import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from tox.constants import ABLATION_SUMMARY, ABLATIONS, CLEAN, SEEDS
from tox.featurize_fp import load_features
from tox.train_aml import run

CONFIGS = [(r, n) for r in (1, 2, 3) for n in (1024, 2048)]


def morgan_all(smiles, radius, size):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
    return np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles]).astype(np.uint8)


def main():
    df = pd.read_parquet(CLEAN)
    _, desc = load_features()
    out = ABLATIONS / "preds"
    out.mkdir(parents=True, exist_ok=True)
    log = {}
    jobs = [(f"xgb_fp-r{r}-{n}", morgan_all(df.smiles, r, n)) for r, n in CONFIGS]
    jobs.append(("xgb_desc", np.zeros((len(df), 0), np.uint8)))
    for name, fp in jobs:
        for seed in SEEDS:
            run(name, "scaffold_det", seed, df, fp, desc, log, out=out)
    (ABLATIONS / "params.json").write_text(json.dumps(log, indent=1))


def summarize(metrics_dir=ABLATIONS / "metrics"):
    """C4 files -> one row per ablation config: 12-task mean per seed, then mean/std over the seeds
    (same recipe as headline.csv, so the two tables are comparable). Full precision, round when displaying."""
    frames = []
    for f in sorted(metrics_dir.glob("*.csv")):
        config, split, seed = f.stem.split("__")
        frames.append(pd.read_csv(f).assign(config=config, split=split, seed=int(seed[4:])))
    df = pd.concat(frames, ignore_index=True)
    per_run = df.groupby(["config", "split", "seed"])[["roc_auc", "pr_auc", "pos_rate"]].mean()
    return (per_run.groupby(["config", "split"])
                   .agg(n_runs=("roc_auc", "size"),
                        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                        pos_rate=("pos_rate", "mean"))
                   .reset_index())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="only write summary.csv from the scored ablations")
    if ap.parse_args().summary:
        summary = summarize()
        summary.to_csv(ABLATION_SUMMARY, index=False)
        print(summary.round(3).to_string(index=False))
    else:
        main()
