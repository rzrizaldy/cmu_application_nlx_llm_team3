#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-$ROOT/../cmu_application_of_nlx_llm/assignment02/.venv/bin/python}"
cd "$ROOT"
for run in T0_generate T1_structured_rag T2_tools T3_guarded; do
  echo "=== $run ==="
  "$PY" experiments/05_team/run_team.py --run "$run" --limit 50
done
echo "=== finetune ==="
"$PY" experiments/05_team/finetune_lora.py
echo "=== T4 ==="
"$PY" experiments/05_team/run_team.py --run T4_finetuned --limit 50 \
  --adapter experiments/05_team/finetune/adapter
"$PY" experiments/05_team/build_chatlogs.py
"$PY" appendix/build_appendix.py
