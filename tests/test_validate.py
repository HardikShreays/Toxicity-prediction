import numpy as np
import pandas as pd
import pytest

from tox.validate import (calibration_table, per_endpoint, reliability, wilcoxon_pair,
                          wilcoxon_table)

TASKS12 = [f"t{j}" for j in range(12)]


def _df(n_seeds=3, shift=0.0, seed=0):
    """C4-shaped frame: two models, one split family, 12 endpoints x n_seeds runs."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.6, 0.9, (12, n_seeds))
    rows = []
    for m, delta in (("a", 0.0), ("b", shift)):
        for j, t in enumerate(TASKS12):
            for s in range(n_seeds):
                rows.append({"model": m, "split": "scaffold", "seed": s, "task": t,
                             "roc_auc": base[j, s] + delta + rng.normal(0, 0.01),
                             "pr_auc": base[j, s] / 2 + delta})
    return pd.DataFrame(rows)


def test_per_endpoint_pairs_models_on_the_same_12_tasks():
    tab = per_endpoint(_df(), "scaffold")
    assert tab.shape == (2, 12)
    assert sorted(tab.columns) == sorted(TASKS12)   # columns are alphabetized; rows share the order
    assert tab.loc["a"].between(0.5, 1.0).all()


def test_wilcoxon_identical_models_is_not_significant_shifted_is():
    same = wilcoxon_pair(_df(), "a", "b", "scaffold")
    assert same["p_value"] > 0.05 and same["n_endpoints"] == 12
    better = wilcoxon_pair(_df(shift=0.05), "a", "b", "scaffold")
    assert better["p_value"] < 0.05
    with pytest.raises(KeyError):
        wilcoxon_pair(_df(), "nope", "b", "scaffold")


def test_wilcoxon_table_skips_missing_families():
    df = _df()
    # model b exists in both families, model a only in scaffold: the random pair is skipped
    a_only = pd.concat([df, df[df.model == "b"].assign(split="random")])
    tab = wilcoxon_table(a_only, models=["a", "b"], families=["scaffold", "random"])
    assert set(tab.family) == {"scaffold"}
    assert len(tab) == 2                             # 1 pair x 2 metrics
    assert (tab.p_value >= 0).all()


def test_reliability_bins_and_ece():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000).astype(float)
    p = np.clip(y * 0.8 + 0.1 + rng.normal(0, 0.05, 1000), 0, 1)
    rel, ece = reliability(y, p)
    assert set(rel.columns) >= {"bin_low", "bin_high", "mean_predicted", "fraction_positive", "count"}
    assert rel.bin_low.is_monotonic_increasing
    assert rel["count"].sum() == 1000
    assert ((rel.fraction_positive >= 0) & (rel.fraction_positive <= 1)).all()
    assert ece >= 0
    # missing labels are excluded
    y2 = y.copy()
    y2[:100] = np.nan
    rel2, _ = reliability(y2, p)
    assert rel2["count"].sum() == 900
    # a perfectly calibrated constant predictor has ~zero ECE in its bin
    _, ece_perf = reliability(np.ones(100), np.full(100, 0.5) * 0 + 1.0)
    assert ece_perf == pytest.approx(0.0, abs=1e-9)


def test_calibration_table_reads_c3_files(tmp_path):
    pytest.importorskip("pyarrow")
    from tox.constants import TASKS as REAL_TASKS
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1"]
    labels = pd.DataFrame({"smiles": smiles}).set_index("smiles")
    labels[REAL_TASKS] = np.nan
    labels.loc[smiles[:4], REAL_TASKS[0]] = [1.0, 0.0, 1.0, 0.0]
    labels.loc[smiles[4:], REAL_TASKS[1]] = [0.0, 1.0]      # a second labelled endpoint
    for m, q in (("a", [0.9, 0.1, 0.8, 0.2]), ("b", [0.5, 0.5, 0.5, 0.5])):
        for seed in (0, 1):
            rows = [{"smiles": s, "role": "test",
                     **{t: (q[i] if i < len(q) else 0.5) for t in REAL_TASKS}}
                    for i, s in enumerate(smiles)]
            pd.DataFrame(rows).to_parquet(tmp_path / f"{m}__scaffold_det__seed{seed}.parquet")
    tab = calibration_table(preds_dir=tmp_path, labels=labels[REAL_TASKS])
    assert set(tab.model) == {"a", "b"}
    a = tab.set_index("model").loc["a"]
    assert a.n_pairs == 12                            # 6 labelled x 2 seeds
    assert a.ece < 0.2                                # confident and roughly right
    assert a.brier < tab.set_index("model").loc["b", "brier"]   # confident+right beats constant 0.5
