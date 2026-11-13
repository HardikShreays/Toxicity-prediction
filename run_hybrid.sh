#!/usr/bin/env bash
# Hybrid stages (plan p. 18-19, Hardik): variants grid -> scoring -> ablations -> analyses ->
# Wilcoxon + calibration -> aggregate + report tables. Resumable: every stage skips outputs
# that already exist. Kept separate from run_all.sh / run_gnn.sh; it only writes
# results/hybrid*, results/ablations/hybrid, results/validation.
# Run AFTER run_all.sh and run_gnn.sh (needs the fingerprint C3 files and the GNN checkpoints).
# usage: PY=/path/to/python ./run_hybrid.sh
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
mkdir -p results/hybrid
LOG=results/hybrid/pipeline.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
fail() { say "FAIL: $*"; echo "$*" > results/hybrid/PIPELINE_FAILED; exit 1; }

if pgrep -f "tox\.(train|ablate)_hybrid|tox\.train_hybrid" >/dev/null; then
  echo "a hybrid job is already running; refusing to start a duplicate" >&2
  exit 2
fi
rm -f results/hybrid/PIPELINE_FAILED results/hybrid/PIPELINE_DONE

# the grid checkpoints the fusion encoder warm-starts from must exist for all RUNS
for split in scaffold_det scaffold_s0 scaffold_s1 scaffold_s2 random_s0 random_s1 random_s2; do
  for seed in 0 1 2; do
    case "$split" in
      scaffold_s0|random_s0) [ "$seed" = 0 ] && continue ;;
      scaffold_s1|random_s1) [ "$seed" = 1 ] && continue ;;
      scaffold_s2|random_s2) [ "$seed" = 2 ] && continue ;;
    esac
    [ -f "results/gnn/mpnn__${split}__seed${seed}.pt" ] || \
      fail "missing results/gnn/mpnn__${split}__seed${seed}.pt; run ./run_gnn.sh first"
  done
done

say "stage1: hybrid variants grid (3 variants x 9 runs)"
"$PY" -W ignore -m tox.train_hybrid --grid > results/hybrid/stage1.log 2>&1 || fail "grid crashed (see results/hybrid/stage1.log)"
"$PY" -W ignore -m tox.train_hybrid --score >> results/hybrid/stage1.log 2>&1 || fail "scoring crashed"
n=$(ls results/preds | grep -cE "^hyb_"); [ "$n" = 27 ] || fail "expected 27 hybrid prediction files, found $n"
say "stage1: OK (27 runs scored)"

say "stage2: hybrid ablations (scaffold_det, 3 seeds, 3 arms)"
"$PY" -W ignore -m tox.train_hybrid --ablations > results/hybrid/stage2.log 2>&1 || fail "ablations crashed (see results/hybrid/stage2.log)"
"$PY" -W ignore -m tox.train_hybrid --score >> results/hybrid/stage2.log 2>&1 || fail "ablation scoring crashed"
"$PY" -W ignore -m tox.train_hybrid --summary >> results/hybrid/stage2.log 2>&1 || fail "ablation summary crashed"
say "stage2: OK (ablations scored, summary written)"

say "stage3: hybrid analyses (sparse/dense endpoints, Tanimoto bins)"
"$PY" -W ignore -m tox.hybrid_analysis >> results/hybrid/stage3.log 2>&1 || fail "analyses crashed (see results/hybrid/stage3.log)"
say "stage3: OK (results/hybrid/sparse_dense.csv, sim_bins.csv)"

say "stage4: technical validation (Wilcoxon, calibration)"
"$PY" -W ignore -m tox.validate --wilcoxon --calibration >> results/hybrid/stage4.log 2>&1 || fail "validation crashed (see results/hybrid/stage4.log)"
say "stage4: OK (results/validation/)"

say "stage5: aggregate + report tables"
"$PY" -W ignore -m tox.aggregate >> results/hybrid/stage5.log 2>&1 || fail "aggregate crashed"
"$PY" -W ignore -m tox.gnn_report >> results/hybrid/stage5.log 2>&1 || fail "gnn_report crashed"
say "stage5: OK (summary.csv, headline.csv, README table, report tables)"
date -u +%FT%TZ > results/hybrid/PIPELINE_DONE
