#!/usr/bin/env bash
# reproduces every phase 1 number: data -> splits -> features -> aml -> metrics -> summary -> figures
# full run takes ~1h on an 8-core laptop (the RF grid is most of it)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

python -m tox.data
python -m tox.splits
python -m tox.featurize_fp
python -m tox.train_aml
python -m tox.ablate_fp
python -m tox.metrics --preds results/preds/ --role test
python -m tox.metrics --preds results/ablations/preds --out results/ablations/metrics --role test
python -m tox.ablate_fp --summary
python -m tox.aggregate

# figures + analysis
for nb in notebooks/01_eda.ipynb notebooks/02_aml.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
