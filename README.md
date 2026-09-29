# Tox21 toxicity prediction

Predicting 12 Tox21 toxicity endpoints from SMILES. Phase 1 = fingerprints/descriptors + classical ML
(LR, RF, XGBoost), evaluated on scaffold and random splits. Phase 2 (GNNs + hybrid) comes later.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Run

```bash
./run_all.sh
```

(WIP - results table goes here once the AML grid is done)
