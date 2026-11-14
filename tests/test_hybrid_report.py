import numpy as np
import pandas as pd

from tox.hybrid_report import (ablation_table, calibration_table, comparison_table,
                               sparse_dense_table, sim_bins_table, wilcoxon_table)


def _headline(models):
    rows = []
    for m in models:
        for s, roc, pr, pos in (("scaffold_det", 0.75, 0.35, 0.10), ("scaffold", 0.72, 0.32, 0.10),
                                ("random", 0.80, 0.42, 0.08)):
            rows.append({"model": m, "split": s, "roc_auc_mean": roc, "roc_auc_std": 0.01,
                         "pr_auc_mean": pr, "pr_auc_std": 0.02, "pos_rate": pos})
    return pd.DataFrame(rows)


def test_comparison_table_lists_aml_gnn_and_hybrids():
    latex = comparison_table(_headline(["mpnn", "hyb_fusion", "rf_fpdesc"]))
    assert r"\label{tab:hybrid-main}" in latex
    for token in ("hyb\\_fusion", "mpnn", "rf\\_fpdesc"):
        assert token in latex
    assert latex.count("\\\\") >= 4                     # 3 model rows + the positive-rate row
    assert "hyb\\_stack" not in latex                   # only present models are listed


def test_ablation_table():
    s = pd.DataFrame({"variant": ["hyb_fusion"] * 2, "arm": ["baseline", "nofp"],
                      "n_runs": [3, 3], "roc_auc_mean": [0.75, 0.73], "roc_auc_std": [0.01, 0.01],
                      "pr_auc_mean": [0.35, 0.33], "pr_auc_std": [0.02, 0.02]})
    latex = ablation_table(s)
    assert r"\label{tab:hybrid-ablation}" in latex
    assert "baseline & 3 & 0.750" in latex and "nofp & 3 & 0.730" in latex


def test_sparse_dense_table():
    sd = pd.DataFrame({"model": ["hyb_fusion"] * 2, "group": ["4 sparsest", "4 densest"],
                       "roc_auc_mean": [0.70, 0.78], "roc_auc_std": [0.01, 0.01],
                       "pr_auc_mean": [0.20, 0.45], "pr_auc_std": [0.02, 0.02]})
    latex = sparse_dense_table(sd)
    assert "sparsest" in latex and "densest" in latex
    assert "0.700" in latex and "0.780" in latex


def test_sim_bins_table():
    sim = pd.DataFrame({"model": ["hyb_fusion"] * 2, "bin": ["[0, 0.3)", "[0.5, 1)"],
                        "n_mols": [120, 900], "roc_auc_mean": [0.60, 0.80],
                        "roc_auc_std": [0.03, np.nan]})
    latex = sim_bins_table(sim)
    assert "[0, 0.3)" in latex and "120" in latex
    assert "0.600 $\\pm$ 0.030" in latex and "0.800 $\\pm$ --" in latex   # NaN std stays "--"


def test_wilcoxon_table_marks_significance():
    w = pd.DataFrame({"family": ["scaffold"] * 2, "metric": ["roc_auc"] * 2,
                      "model_a": ["hyb_fusion"] * 2, "model_b": ["mpnn"] * 2,
                      "roc_auc_a": [0.75, 0.75], "roc_auc_b": [0.73, 0.73],
                      "p_value": [0.04, 0.20]})
    latex = wilcoxon_table(w)
    assert "0.040$^{*}$" in latex                       # p < 0.05 marked
    assert "0.200 \\\\" in latex                        # p >= 0.05 unmarked


def test_calibration_table():
    cal = pd.DataFrame({"model": ["mpnn"], "n_pairs": [170000], "ece": [0.081], "brier": [0.12]})
    latex = calibration_table(cal)
    assert "170,000 & 0.081 & 0.120" in latex


def test_sparse_dense_grouping():
    from tox.hybrid_analysis import sparse_dense
    rows = []
    for j, n_pos in enumerate([10, 20, 30, 40, 500, 600, 700, 800] * 2):
        rows.append({"model": "mpnn", "seed": 0, "task": f"t{j}", "n_pos": n_pos,
                     "roc_auc": 0.7, "pr_auc": 0.3})
    out = sparse_dense(pd.DataFrame(rows), k=4)
    assert set(out.group) == {"4 sparsest", "4 densest"}
    assert out.set_index("group").roc_auc_mean["4 sparsest"] == 0.7
    assert len(out) == 2
