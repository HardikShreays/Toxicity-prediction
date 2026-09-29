"""S4 - scaffold and random splits (contract C2), all 80/10/10.

scaffold_det   : deepchem convention, biggest scaffold groups first
scaffold_s{k}  : balanced random scaffold split, seed k
random_s{k}    : plain random split by molecule, seed k (reference only)

Acyclic molecules all get scaffold "" and form one big group -> they land in train.
"""
import json

import numpy as np
import pandas as pd
import yaml

from tox.constants import CLEAN, CONFIGS, SEEDS, SPLITS, TASKS

CFG = yaml.safe_load((CONFIGS / "split.yaml").read_text())
FRAC_TRAIN, FRAC_VAL = CFG["frac_train"], CFG["frac_val"]


def _groups(scaffolds):
    """scaffold -> list of row indices, in first-seen order."""
    return list(pd.Series(range(len(scaffolds))).groupby(list(scaffolds), sort=False).apply(list))


def _fill(groups, n):
    """Greedy: groups go to train until it's full, then val, then the rest to test."""
    train, val, test = [], [], []
    n_train, n_val = FRAC_TRAIN * n, FRAC_VAL * n
    for g in groups:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(val) + len(g) <= n_val:
            val += g
        else:
            test += g
    return train, val, test


def scaffold_det(scaffolds):
    groups = sorted(_groups(scaffolds), key=lambda g: (len(g), -g[0]), reverse=True)
    return _fill(groups, len(scaffolds))


def scaffold_balanced(scaffolds, seed):
    n = len(scaffolds)
    groups = _groups(scaffolds)
    big_cut = (1 - FRAC_TRAIN - FRAC_VAL) * n / 2          # half the test size
    big = [g for g in groups if len(g) > big_cut]
    small = [g for g in groups if len(g) <= big_cut]
    rng = np.random.default_rng(seed)
    small = [small[i] for i in rng.permutation(len(small))]
    return _fill(big + small, n)


def random_split(n, seed):
    idx = np.random.default_rng(seed).permutation(n).tolist()
    a, b = int(FRAC_TRAIN * n), int((FRAC_TRAIN + FRAC_VAL) * n)
    return idx[:a], idx[a:b], idx[b:]


def check_positives(df, parts, name):
    """Every task should have >=1 positive in val and test. Warn loudly, never hand-move molecules."""
    for role in ("val", "test"):
        y = df.iloc[parts[role]][TASKS]
        missing = [t for t in TASKS if (y[t] == 1).sum() == 0]
        if missing:
            print(f"WARNING {name}: no positives in {role} for {missing}")


def to_json(df, parts, seed):
    out = {k: df.smiles.iloc[v].tolist() for k, v in parts.items()}
    out["meta"] = {
        "seed": seed,
        "sizes": {k: len(v) for k, v in parts.items()},
        "n_scaffolds": {k: int(df.scaffold.iloc[v].nunique()) for k, v in parts.items()},
    }
    return out


def make_all(df):
    s = df.scaffold.tolist()
    splits = {"scaffold_det": (scaffold_det(s), None)}
    for seed in SEEDS:
        splits[f"scaffold_s{seed}"] = (scaffold_balanced(s, seed), seed)
        splits[f"random_s{seed}"] = (random_split(len(df), seed), seed)
    return {name: (dict(zip(["train", "val", "test"], p)), seed) for name, (p, seed) in splits.items()}


def load(name):
    return json.loads((SPLITS / f"{name}.json").read_text())


def main():
    df = pd.read_parquet(CLEAN)
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, (parts, seed) in make_all(df).items():
        check_positives(df, parts, name)
        js = to_json(df, parts, seed)
        (SPLITS / f"{name}.json").write_text(json.dumps(js))
        print(name, js["meta"])


if __name__ == "__main__":
    main()
