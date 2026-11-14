"""Hardik - LaTeX tables for the Phase 2 hybrid / comparison report sections.

Nothing here computes a metric: every table is generated from the shared scorer's output
(tox.aggregate.load_all), the hybrid ablation summary (tox.train_hybrid.summarize) and the
validation tables (tox.validate) - the same numbers the CSVs in results/ carry.

    python -m tox.hybrid_report        # -> report/phase2/hybrid_results_tables.tex
"""
from __future__ import annotations

import pandas as pd

from tox.aggregate import load_all
from tox.constants import ABLATIONS, HYBRID, RESULTS, ROOT, VALIDATION
from tox.gnn_report import FAMILIES, headline_frame

OUT = ROOT / "report" / "phase2" / "hybrid_results_tables.tex"

HYBRIDS = ["hyb_fusion", "hyb_embxgb", "hyb_stack"]
GNN = ["gcn", "mpnn", "afp"]
AML = ["lr_fp", "lr_fpdesc", "rf_fp", "rf_fpdesc", "xgb_fp", "xgb_fpdesc"]


def _cell(g: pd.DataFrame, split: str, key: str) -> str:
    if split not in g.index:
        return "--"
    r = g.loc[split]
    std = "--" if pd.isna(r[key + "_std"]) else f"{r[key + '_std']:.3f}"
    return f"{r[key + '_mean']:.3f} $\\pm$ {std}"


def comparison_table(headline: pd.DataFrame) -> str:
    """Final comparison: AML + GNN + hybrids on the three split families (test, 12-endpoint mean)."""
    models = [m for m in AML + GNN + HYBRIDS if m in set(headline.model)]
    rows = []
    for m in models:
        g = headline[headline.model == m].set_index("split")
        rows.append(f"{m.replace('_', chr(92) + '_')} & "
                    + " & ".join(_cell(g, s, "roc_auc") for s in FAMILIES)
                    + " & " + " & ".join(_cell(g, s, "pr_auc") for s in FAMILIES) + r" \\")
    base = headline.groupby("split").pos_rate.mean()
    base_row = ("positive rate (random PR-AUC) & \\multicolumn{3}{c}{} & "
                + " & ".join(f"{base[s]:.3f}" if s in base else "--" for s in FAMILIES) + r" \\")
    return "\n".join([
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Final Phase 2 comparison: test ROC-AUC and PR-AUC, 12-endpoint mean, "
        r"mean $\pm$ std over 3 runs. All models share the splits, the fingerprints and the "
        r"scorer; the hybrids differ only in how the two parents are combined.}",
        r"\label{tab:hybrid-main}", r"\small", r"\begin{tabular}{lcccccc}", r"\toprule",
        r" & \multicolumn{3}{c}{ROC-AUC} & \multicolumn{3}{c}{PR-AUC} \\",
        r"model & scaffold\_det & scaffold & random & scaffold\_det & scaffold & random \\", r"\midrule",
        *rows, r"\midrule", base_row, r"\bottomrule", r"\end{tabular}", r"\end{table*}"])


def ablation_table(summary: pd.DataFrame) -> str:
    """Hybrid ablation arms (results/ablations/hybrid/summary.csv): one row per arm."""
    rows = []
    for _, r in summary.iterrows():
        std = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
        rows.append(f"{r.variant.replace('_', chr(92) + '_')} & "
                    f"{r.arm.replace('_', chr(92) + '_')} & {int(r.n_runs)} & "
                    f"{r.roc_auc_mean:.3f} $\\pm$ {std(r.roc_auc_std)} & "
                    f"{r.pr_auc_mean:.3f} $\\pm$ {std(r.pr_auc_std)} \\\\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Hybrid ablations on \texttt{scaffold\_det}: test ROC-AUC and PR-AUC "
        r"(12-endpoint mean, mean $\pm$ std over seeds). One factor changes per arm; the "
        r"baseline is the main run of the same variant.}",
        r"\label{tab:hybrid-ablation}", r"\small",
        r"\begin{tabular}{llccc}", r"\toprule",
        r"variant & arm & runs & ROC-AUC & PR-AUC \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def sparse_dense_table(sd: pd.DataFrame) -> str:
    """Hybrid vs parents on the 4 sparsest vs 4 densest endpoints (results/hybrid/sparse_dense.csv)."""
    rows = []
    for _, r in sd.iterrows():
        std = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
        rows.append(f"{r.model.replace('_', chr(92) + '_')} & {r.group} & "
                    f"{r.roc_auc_mean:.3f} $\\pm$ {std(r.roc_auc_std)} & "
                    f"{r.pr_auc_mean:.3f} $\\pm$ {std(r.pr_auc_std)} \\\\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Test ROC-AUC and PR-AUC on the 4 sparsest vs the 4 densest endpoints "
        r"(\texttt{scaffold\_det}), the plan's direct test of the hybrid rationale: combining "
        r"two parents should help most where labels are rare.}",
        r"\label{tab:hybrid-sparse}", r"\small",
        r"\begin{tabular}{llcc}", r"\toprule",
        r"model & endpoints & ROC-AUC & PR-AUC \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def sim_bins_table(sim: pd.DataFrame) -> str:
    """ROC-AUC by Tanimoto-similarity bin (results/hybrid/sim_bins.csv)."""
    rows = []
    for _, r in sim.iterrows():
        std = "--" if pd.isna(r.roc_auc_std) else f"{r.roc_auc_std:.3f}"
        rows.append(f"{r.model.replace('_', chr(92) + '_')} & {r.bin} & {int(r.n_mols)} & "
                    f"{r.roc_auc_mean:.3f} $\\pm$ {std} \\\\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Test ROC-AUC (all 12 tasks pooled, missing labels masked) by maximum Tanimoto "
        r"similarity of each test molecule to the training set (\texttt{scaffold\_det}, mean "
        r"$\pm$ std over seeds). Low-similarity bins are where unseen scaffolds live.}",
        r"\label{tab:hybrid-simbins}", r"\small",
        r"\begin{tabular}{llcc}", r"\toprule",
        r"model & bin & mols & ROC-AUC \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def wilcoxon_table(w: pd.DataFrame) -> str:
    """The scaffold-family Wilcoxon pairs that the report text cites (scaffold vs hybrids)."""
    w = w[(w.family == "scaffold") & w.model_a.isin(HYBRIDS + ["mpnn", "rf_fpdesc", "xgb_fpdesc"])
          | w.model_b.isin(HYBRIDS + ["mpnn", "rf_fpdesc", "xgb_fpdesc"])]
    rows = []
    for _, r in w[w.metric == "roc_auc"].iterrows():
        stars = "$^{**}$" if r.p_value < 0.01 else ("$^{*}$" if r.p_value < 0.05 else "")
        rows.append(f"{r.model_a.replace('_', chr(92) + '_')} & {r.model_b.replace('_', chr(92) + '_')} & "
                    f"{r.roc_auc_a:.3f} & {r.roc_auc_b:.3f} & {r.p_value:.3f}{stars} \\\\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Wilcoxon signed-rank tests on the randomized scaffold splits (12 paired "
        r"endpoint means; each endpoint averaged over its 3 runs). $^{*}$/$^{**}$: $p<0.05$ "
        r"/ $p<0.01$. Exploratory: 12 endpoints and many pairs, no multiplicity correction.}",
        r"\label{tab:hybrid-wilcoxon}", r"\small",
        r"\begin{tabular}{llccc}", r"\toprule",
        r"model $a$ & model $b$ & ROC-AUC $a$ & ROC-AUC $b$ & $p$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def calibration_table(cal: pd.DataFrame) -> str:
    """ECE / Brier per model (results/validation/calibration.csv)."""
    rows = []
    for _, r in cal.iterrows():
        rows.append(f"{r.model.replace('_', chr(92) + '_')} & {int(r.n_pairs):,} & "
                    f"{r.ece:.3f} & {r.brier:.3f} \\\\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Test calibration over all labelled (task, molecule) pairs of every run: "
        r"expected calibration error (10 equal-width bins) and Brier score.}",
        r"\label{tab:hybrid-calibration}", r"\small",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"model & pairs & ECE & Brier \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main():
    parts = ["% GENERATED by python -m tox.hybrid_report from results/ files (no new metrics)"]
    headline = headline_frame(load_all())
    parts.append(comparison_table(headline))
    summ = ABLATIONS / "hybrid" / "summary.csv"
    if summ.exists():
        parts.append(ablation_table(pd.read_csv(summ)))
    sd = HYBRID / "sparse_dense.csv"
    if sd.exists():
        parts.append(sparse_dense_table(pd.read_csv(sd)))
    sim = HYBRID / "sim_bins.csv"
    if sim.exists():
        parts.append(sim_bins_table(pd.read_csv(sim)))
    w = VALIDATION / "wilcoxon.csv"
    if w.exists():
        parts.append(wilcoxon_table(pd.read_csv(w)))
    cal = VALIDATION / "calibration.csv"
    if cal.exists():
        parts.append(calibration_table(pd.read_csv(cal)))
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
