#!/usr/bin/env bash
# reproduces every phase 1 number: data -> splits -> aml -> metrics -> summary
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

python -m tox.data
python -m tox.splits
python -m tox.featurize_fp
python -m tox.train_aml
python -m tox.metrics --preds results/preds/ --role test
python -m tox.aggregate
