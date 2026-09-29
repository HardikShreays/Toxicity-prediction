"""A3 (feature-engineering rubric line) - fingerprint ablation with XGBoost on scaffold_det, 3 seeds:
radius 1/2/3 x 1024/2048 bits, plus descriptors only. Everything else is the same as xgb_fp.
Preds -> results/ablations/preds, then score with
python -m tox.metrics --preds results/ablations/preds --out results/ablations/metrics
"""
import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from tox.constants import ABLATIONS, CLEAN, SEEDS
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


if __name__ == "__main__":
    main()
