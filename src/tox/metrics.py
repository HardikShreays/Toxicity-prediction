"""H2 - the single scorer. Nobody computes final metrics anywhere else.

CLI: python -m tox.metrics --preds results/preds/ --role test
scores every C3 prediction file in the folder and writes one C4 csv per file.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from tox.constants import CLEAN, METRICS, TASKS


def score(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, t in enumerate(TASKS):
        m = ~np.isnan(y_true[:, j])                 # missing labels excluded
        yt, yp = y_true[m, j], y_prob[m, j]
        n_pos = int(yt.sum())
        if len(np.unique(yt)) < 2:                  # degenerate endpoint
            rows.append(dict(task=t, n=len(yt), n_pos=n_pos, pos_rate=np.nan,
                             roc_auc=np.nan, pr_auc=np.nan))
            continue
        rows.append(dict(task=t, n=len(yt), n_pos=n_pos, pos_rate=yt.mean(),
                         roc_auc=roc_auc_score(yt, yp),
                         pr_auc=average_precision_score(yt, yp)))
    return pd.DataFrame(rows)


def load_preds(path: Path, role: str, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Read a C3 file and line its rows up with the true labels by smiles."""
    p = pd.read_parquet(path)
    assert list(p.columns) == ["smiles", "role"] + TASKS, f"{path.name} breaks contract C3"
    p = p[p.role == role]
    probs = p[TASKS].to_numpy(dtype=float)
    assert np.all((probs >= 0) & (probs <= 1)), f"{path.name} has probs outside [0, 1]"
    y = labels.loc[p.smiles, TASKS].to_numpy(dtype=float)
    return y, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--role", choices=["val", "test"], default="test")
    ap.add_argument("--out", type=Path, default=METRICS)
    args = ap.parse_args()

    labels = pd.read_parquet(CLEAN).set_index("smiles")
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.preds.glob("*.parquet"))
    for f in files:
        y, p = load_preds(f, args.role, labels)
        score(y, p).to_csv(args.out / f"{f.stem}.csv", index=False)
    print(f"scored {len(files)} files ({args.role}) -> {args.out}")


if __name__ == "__main__":
    main()
