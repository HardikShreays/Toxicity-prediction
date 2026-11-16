# Phase 1: known gaps and checks not done

Written during the pre-submission audit (Phase 1 only). Nothing here was invented to fill a gap.

## Missing deliverables (owners decide)

- **Error analysis (Priyansh).** The plan's Phase 1 slide arc ends with "error analysis" and P3 lists an
  error gallery (the most confident false positives / false negatives of the best model, drawn as
  structures). Neither the report, the slides, the notebooks nor any remote branch contains one. At the time of
  the audit every teammate branch was already merged into `main` and there were no open pull requests or
  issues for it. Not produced here: it is Priyansh's deliverable and would need his choice of model and examples.
## Resolved since the audit

- ~~**Unit tests for cleaning and dedup (Saman).**~~ `tests/test_data.py` now holds 21 tests covering
  `standardize` (canonicalisation, largest-fragment salt stripping, `None` on unparseable input, charges
  and stereochemistry preserved), `scaffold`, the three dedup rules, and the C1 contract on the committed
  parquet. Two behaviours were pinned down while writing them and are worth knowing: stripping a salt does
  **not** neutralize what is left, so the sodium salt and the free acid stay different molecules and are
  not merged by dedup; and an empty SMILES survives `standardize` (RDKit parses `""` into an empty
  molecule) and then makes `scaffold` raise. Tox21 contains no such row, so the pipeline is unaffected,
  and a test asserts that.
- ~~**Report build.**~~ `report/phase1/main.pdf` has been rebuilt from `main.tex` with `tectonic`, so the
  PDF and its source now agree (this was the "recompile on Overleaf before submitting" item: the two
  corrections it was missing, the heavy-atom wording and NR-ER 0.693, are now in the PDF).

## Not verified

- A fresh Colab runtime (the plan's P0 gate) and Python 3.11 (the version the README names). The clean-clone
  checks ran on macOS arm64 with Python 3.13 and the pinned requirements; with them the 54 AML runs reproduced
  the committed metrics exactly (C4 identical, C3 probabilities identical up to 4.4e-16).
- `brew install libomp` itself (the macOS note in the README follows from XGBoost's library search path).
- The mock viva (a team activity; there is no Phase 1 viva document in the repository).
