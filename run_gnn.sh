#!/usr/bin/env bash
# GNN stages (B4/B5): tuning -> frozen-config grid -> scoring -> ablations -> summary.
# Idempotent and resumable: tune_gnn skips saved configs, the grid and ablations skip existing C3 files.
# Kept separate from run_all.sh (shared file); it only writes GNN outputs (results/gnn*, results/ablations/gnn).
# usage: PY=/path/to/python ./run_gnn.sh      (PY defaults to .venv/bin/python, then python3)
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
mkdir -p results/gnn
LOG=results/gnn/pipeline.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
fail() { say "FAIL: $*"; echo "$*" > results/gnn/PIPELINE_FAILED; exit 1; }

# never run two copies: they would write the same params / prediction files
if pgrep -f "tox\.(tune|train|ablate)_gnn" >/dev/null; then
  echo "a GNN job is already running (pgrep matched tox.*_gnn); refusing to start a duplicate" >&2
  exit 2
fi
rm -f results/gnn/PIPELINE_FAILED results/gnn/PIPELINE_DONE

check_tuning() { "$PY" -W ignore - <<'P'
import json, sys
s = json.load(open("results/gnn_params.json"))
ok = all(m in s and len(s[m]["trials"]) == 12 and "best" in s[m] for m in ("afp", "gcn", "mpnn"))
print({m: len(s.get(m, {}).get("trials", [])) for m in ("afp", "gcn", "mpnn")}); sys.exit(0 if ok else 1)
P
}

say "stage1: tuning (36 configs, validation only)"
if ! check_tuning 2>/dev/null; then
  "$PY" -W ignore -m tox.tune_gnn >> results/gnn/tune_stage1.log 2>&1 || fail "tuning crashed (see results/gnn/tune_stage1.log)"
  check_tuning || fail "tuning still incomplete after resume"
fi
say "stage1: OK (36/36)"

say "stage2: frozen-config grid (27 runs)"
"$PY" -W ignore -m tox.train_gnn --grid > results/gnn/stage2.log 2>&1 || fail "grid crashed (see results/gnn/stage2.log)"
n=$(ls results/preds | grep -cE "^(gcn|mpnn|afp)__"); [ "$n" = 27 ] || fail "expected 27 GNN prediction files, found $n"
"$PY" -W ignore -m tox.train_gnn --score >> results/gnn/stage2.log 2>&1 || fail "test scoring crashed"
"$PY" -W ignore -c "
from tox.train_gnn import score_preds
from tox.constants import RESULTS
print(len(score_preds(role='val', out_dir=RESULTS/'gnn'/'val_metrics')), 'val C4 files')" >> results/gnn/stage2.log 2>&1 || fail "val scoring crashed"
say "stage2: OK (27 runs scored)"

say "stage3: MPNN ablations (27 runs)"
"$PY" -W ignore -m tox.ablate_gnn --model mpnn > results/gnn/stage3.log 2>&1 || fail "ablations crashed (see results/gnn/stage3.log)"
"$PY" -W ignore - >> results/gnn/stage3.log 2>&1 <<'P' || fail "ablation scoring/summary crashed"
from tox.ablate_gnn import OUT, variants, summarize, per_task_summary
from tox.train_gnn import score_preds
tok = set(variants("mpnn"))
a = score_preds(OUT / "preds", OUT / "metrics", "test", tok)
b = score_preds(OUT / "preds", OUT / "val_metrics", "val", tok)
print(len(a), len(b), "ablation C4 files (test, val)")
summarize("mpnn").to_csv(OUT / "summary.csv", index=False)
per_task_summary("mpnn").to_csv(OUT / "per_task.csv", index=False)
P
n=$(ls results/ablations/gnn/preds | wc -l); [ "$n" = 27 ] || fail "expected 27 ablation prediction files, found $n"
say "stage3: OK (27 ablation runs scored, summary written)"
date -u +%FT%TZ > results/gnn/PIPELINE_DONE
