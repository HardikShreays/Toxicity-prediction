import re

import numpy as np
import pandas as pd
import pytest

from tox.gnn_report import ablation_table, headline_frame, main_table


def _df():
    rows = []
    rng = np.random.default_rng(0)
    for model in ("gcn", "xgb_fp", "unrelated_model"):
        for split, seeds in (("scaffold_det", [0, 1, 2]), ("scaffold", [0, 1, 2]), ("random", [0, 1, 2])):
            for s in seeds:
                for t in range(12):
                    rows.append(dict(model=model, split=split, seed=s, task=t, roc_auc=0.7 + 0.01 * s,
                                     pr_auc=0.3, pos_rate=0.1 if split != "random" else 0.08))
    return pd.DataFrame(rows)


def test_headline_recipe_matches_aggregate_definition():
    h = headline_frame(_df()).set_index(["model", "split"])
    assert h.loc[("gcn", "scaffold_det"), "n_runs"] == 3
    assert h.loc[("gcn", "scaffold_det"), "roc_auc_mean"] == pytest.approx(0.71)
    assert h.loc[("gcn", "scaffold_det"), "roc_auc_std"] == pytest.approx(np.std([0.70, 0.71, 0.72], ddof=1))


def test_main_table_lists_gnn_first_and_has_no_best_marker():
    tex = main_table(headline_frame(_df()))
    assert tex.index("gcn &") < tex.index("xgb\\_fp &") and "unrelated" not in tex
    assert "0.710 $\\pm$ 0.010" in tex and "positive rate" in tex
    assert "\\textbf" not in tex                                        # no row is highlighted as 'best'
    assert tex.count("\\begin{") == tex.count("\\end{") and tex.count("{") == tex.count("}")


def test_main_table_handles_missing_split_and_single_run():
    h = headline_frame(_df()[lambda d: ~((d.model == "gcn") & (d.split == "random"))])
    tex = main_table(h)
    gcn_line = [l for l in tex.splitlines() if l.startswith("gcn &")][0]
    assert gcn_line.count("--") == 2                                   # ROC and PR for the absent split
    one = _df()[lambda d: (d.seed == 0)]
    assert "--" in main_table(headline_frame(one))                     # std undefined for a single run


def test_ablation_table_rows_and_balance():
    s = pd.DataFrame([
        dict(group="baseline", variant="mpnn", n_runs=3, roc_auc_mean=.75, roc_auc_std=.01, pr_auc_mean=.34, pr_auc_std=.01),
        dict(group="bond", variant="mpnn-noedge", n_runs=3, roc_auc_mean=.74, roc_auc_std=.02, pr_auc_mean=.33, pr_auc_std=.02)])
    tex = ablation_table(s)
    assert "baseline & mpnn & 3 & 0.750 $\\pm$ 0.010" in tex and "mpnn-noedge" in tex
    assert tex.count("{") == tex.count("}") and tex.count("\\begin{") == tex.count("\\end{")
