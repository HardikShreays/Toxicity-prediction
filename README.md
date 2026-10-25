# Tox21 toxicity prediction

Predicting 12 Tox21 toxicity endpoints from SMILES, and checking how much performance survives on
**unseen chemical scaffolds**. Phase 1 = Morgan fingerprints + RDKit descriptors with
logistic regression, random forest and XGBoost. Phase 2 adds GNNs (and a hybrid model, in progress) on the same
splits and scorer.

## Setup

Python 3.11.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # 29 tests: metrics, splits, featurization, xgb importance
                            # (the 7 real-split checks skip until tox.data has built the parquet)
```

## Reproduce everything

```bash
./run_all.sh
```

Downloads the data, cleans + dedups it, builds the 7 splits, computes features, trains the whole
AML grid (6 configs x 12 endpoints x 9 runs) and the fingerprint ablation, scores everything with
`tox.metrics`, writes `results/summary.csv` + `results/headline.csv`, refreshes the table below and
re-runs the notebooks. Takes about 1 hour on an 8-core M3 laptop, most of it the random forests.
`train_aml.py` skips runs whose prediction file already exists, so it can be resumed.

### GNN pipeline (Phase 2)

```bash
PY=/path/to/python ./run_gnn.sh     # PY defaults to .venv/bin/python, then python3
```

Run after `run_all.sh` (needs its splits). Three stages: tune GCN / MPNN / AttentiveFP on the
`scaffold_det` validation set (36 configs), train the frozen-config grid (3 models x 3 splits x 3 seeds
= 27 runs) and score it, then the MPNN ablations (27 runs). It is resumable and refuses to start if a
GNN job is already running. Logs are in `results/gnn/pipeline.log`; `PIPELINE_DONE` / `PIPELINE_FAILED`
mark the outcome. Needs PyTorch + PyTorch Geometric (not in `requirements.txt`).

## Results (test, 12-endpoint mean, mean ± std over 3 runs)

Auto-filled by `python -m tox.aggregate`. Mean test positive rate (= PR-AUC of a random classifier)
is 0.103 on the scaffold splits and 0.077 on random, so compare PR-AUC within a column only.

<!-- results:start -->
| model | ROC scaffold_det | ROC scaffold | ROC random | PR scaffold_det | PR scaffold | PR random |
|---|---|---|---|---|---|---|
| lr_fp | 0.715 ± 0.000 | 0.775 ± 0.036 | 0.801 ± 0.013 | 0.319 ± 0.000 | 0.395 ± 0.051 | 0.435 ± 0.038 |
| lr_fpdesc | 0.722 ± 0.000 | 0.788 ± 0.039 | 0.826 ± 0.010 | 0.299 ± 0.000 | 0.367 ± 0.068 | 0.463 ± 0.029 |
| rf_fp | 0.732 ± 0.001 | 0.803 ± 0.019 | 0.833 ± 0.009 | 0.366 ± 0.001 | 0.442 ± 0.043 | 0.484 ± 0.019 |
| rf_fpdesc | 0.782 ± 0.003 | 0.839 ± 0.024 | 0.862 ± 0.002 | 0.406 ± 0.002 | 0.481 ± 0.043 | 0.531 ± 0.016 |
| xgb_fp | 0.696 ± 0.003 | 0.767 ± 0.045 | 0.801 ± 0.012 | 0.306 ± 0.006 | 0.389 ± 0.047 | 0.435 ± 0.024 |
| xgb_fpdesc | 0.761 ± 0.008 | 0.813 ± 0.032 | 0.855 ± 0.007 | 0.391 ± 0.008 | 0.446 ± 0.057 | 0.520 ± 0.020 |
<!-- results:end -->

Model selection is on validation PR-AUC only: `rf_fpdesc` on `scaffold_det`, `xgb_fpdesc` on the
other two. Per-endpoint numbers are in `results/summary.csv`, analysis in `notebooks/02_aml.ipynb`.

## Layout

```
src/tox/
  constants.py      TASKS, SEEDS, every path
  data.py           download, standardize (largest fragment), dedup   -> C1
  splits.py         scaffold_det, scaffold_s{0,1,2}, random_s{0,1,2}  -> C2
  featurize_fp.py   Morgan r2/2048 + 217 RDKit descriptors (cached)
  models_aml.py     LR / RF / XGB + train-only descriptor pipeline
  train_aml.py      full grid, tuned on val PR-AUC                     -> C3
  ablate_fp.py      radius / length / descriptors-only ablation
  metrics.py        the only scorer (masked ROC-AUC / PR-AUC)          -> C4
  aggregate.py      summary.csv, headline.csv, README table
  featurize_graph.py  SMILES -> PyG molecular graphs
  models_gnn.py     GCN / MPNN / AttentiveFP
  train_gnn.py      masked-loss training, early stopping, C3 export
  tune_gnn.py       hyper-parameter search on val only
  ablate_gnn.py     MPNN architecture ablations
  gnn_report.py     LaTeX tables from saved C4 files (no new metrics)
  analysis/         importance.py (gain -> substructures, figures), make_slides_phase1.py
notebooks/          01_eda, 02_aml
report/phase1/      IEEEtran LaTeX report, slides.pptx (built by analysis/make_slides_phase1.py)
```

## Data contracts

| ID | path | format |
|---|---|---|
| C1 | `data/processed/tox21_clean.parquet` | `mol_id, smiles, scaffold` + 12 task columns (1.0 / 0.0 / NaN) |
| C2 | `data/splits/<name>.json` | `{"train": [smiles], "val": [...], "test": [...], "meta": {...}}` |
| C3 | `results/preds/{model}__{split}__seed{s}.parquet` | `smiles, role` + 12 probability columns |
| C4 | `results/metrics/{model}__{split}__seed{s}.csv` | `task, n, n_pos, pos_rate, roc_auc, pr_auc` |

C1 and the feature caches are git-ignored and rebuilt by `run_all.sh`; the split files, predictions and
metrics are committed.

## Decisions worth knowing

- Duplicates are merged on canonical SMILES **before** splitting; conflicting labels become NaN (44 cases).
- Charges and stereochemistry are kept; only salts/counter-ions are stripped.
- Missing labels are never filled: each endpoint trains on its labelled rows, and the scorer masks NaNs.
- Everything fitted (imputer, scaler, class weights, `scale_pos_weight`) sees train rows only.
- `Ipc` is log-transformed because raw values (~1e41) overflow float32 in the tree models.
- Validation is reused: hyper-parameters, the reported model and XGBoost early stopping are all chosen on it,
  so val scores are optimistic (33/216 XGBoost fits stop at <= 20 trees). Test is never used for any choice.
