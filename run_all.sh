#!/usr/bin/env bash
# Reproduces every number in BOTH reports (plan gate P6).
#   ./run_all.sh            phase 1, then the GNN stages, then the hybrid stages
#   ./run_all.sh --phase1   phase 1 only (data -> splits -> features -> aml -> metrics -> figures)
# Phase 1 takes ~1h on an 8-core laptop (the RF grid is most of it); the GNN and hybrid stages
# add several hours on CPU. Every stage is idempotent, so a re-run resumes rather than repeats.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

PHASE1_ONLY=0
[ "${1:-}" = "--phase1" ] && PHASE1_ONLY=1

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
python -m tox.analysis.importance
python -m tox.analysis.make_slides_phase1
for nb in notebooks/01_eda.ipynb notebooks/02_aml.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done

[ "$PHASE1_ONLY" = 1 ] && { echo "phase 1 done (--phase1); skipping the GNN and hybrid stages"; exit 0; }

# phase 2. Separate scripts because each one locks against duplicate runs and keeps its own
# resumable stage log; run in this order - the hybrid needs the fingerprint C3 files above and
# the GNN checkpoints from run_gnn.sh.
./run_gnn.sh
./run_hybrid.sh

# interpretability (plan P3): reuses the checkpoints run_gnn.sh leaves in results/gnn/ckpt,
# so this is a few seconds here and only retrains if those have been cleaned away
python -m tox.analysis.interpret_gnn --model mpnn --task SR-MMP

# final roll-up over phase 1 + phase 2, so summary.csv / headline.csv cover every model
python -m tox.aggregate
