# Phase 1: known gaps and checks not done

Written during the pre-submission audit (Phase 1 only). Nothing here was invented to fill a gap.

## Missing deliverables (owners decide)

- **Error analysis (Priyansh).** The plan's Phase 1 slide arc ends with "error analysis" and P3 lists an
  error gallery (the most confident false positives / false negatives of the best model, drawn as
  structures). Neither the report, the slides, the notebooks nor any remote branch contains one. At the time of
  the audit every teammate branch was already merged into `main` and there were no open pull requests or
  issues for it. Not produced here: it is Priyansh's deliverable and would need his choice of model and examples.
- **Unit tests for cleaning and dedup (Saman).** `tests/test_data.py` is a 0-byte file on `main`
  (commit a966885, message "S3: tests for standardize, dedup rules and the C1 contract"). No test imports
  `tox.data`. The cleaned data, dedup rules and C1 contract were checked independently during the audit
  (columns, canonical unique SMILES, scaffolds, dedup counts, split coverage), but no committed test
  guards them.

## Report build

- `report/phase1/main.pdf` was built before two small corrections to `main.tex` (the heavy-atom wording
  "100 or more", and NR-ER ROC-AUC 0.693 instead of 0.692). No LaTeX toolchain was available on the audit
  machine, so the PDF was not rebuilt: **recompile on Overleaf before submitting** so PDF and source agree.
  The committed PDF was otherwise checked against its source: same title, sections and numbers.

## Not verified

- A fresh Colab runtime (the plan's P0 gate) and Python 3.11 (the version the README names). The clean-clone
  checks ran on macOS arm64 with Python 3.13 and the pinned requirements; with them the 54 AML runs reproduced
  the committed metrics exactly (C4 identical, C3 probabilities identical up to 4.4e-16).
- `brew install libomp` itself (the macOS note in the README follows from XGBoost's library search path).
- The mock viva (a team activity; there is no Phase 1 viva document in the repository).
