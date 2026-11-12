#!/bin/bash
# The last job: a summary of every sweep and one archive with everything Theo needs.
# Submitted by run_all.sh after all the other jobs; needs no GPU.
#   results/summary_<time>.txt              what finished, what failed, what was scored
#   results/disensa_results_<time>.tar.gz   runs/ of every experiment sweep + logs/ + summary
#   pack_job.sh followup: the follow-up sweeps only (configs/experiments/followup/), in
#   results/disensa_followup_<time>.tar.gz, so the archive holds just the new runs

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit with scripts/indaco/run_all.sh}"
cd "$REPO"
# shellcheck source=/dev/null
source scripts/indaco/cluster.env
PY="$REPO/.venv/bin/python"
export LITELLM_LOCAL_MODEL_COST_MAP=True
shopt -s nullglob

stamp="$(date +%Y%m%d-%H%M)"
mkdir -p results
if [[ "${1:-}" == followup ]]; then
  sweeps=(configs/experiments/followup/*.yaml)
  kind=followup
else
  sweeps=(configs/experiments/pilot/*.yaml configs/experiments/core/*.yaml configs/experiments/main/*.yaml)
  kind=results
fi
summary="results/summary_${kind}_$stamp.txt"
{
  echo "disensa results, $(date "+%Y-%m-%d %H:%M"), commit $(git rev-parse --short HEAD)" \
    "$(git diff --quiet || echo '(with uncommitted changes!)')"
  echo
  echo "== per sweep (TOTAL runs, FINISHED, FAILED = stopped by the failure guard, SCORED by $SCORER)"
  for f in "${sweeps[@]}"; do
    "$PY" scripts/indaco/preflight.py state "$f" --scorer "$SCORER"
  done
  echo
  folders=()
  for f in "${sweeps[@]}"; do
    name="$(basename "$(dirname "$f")")"
    stem="$(basename "$f" .yaml)"
    [[ "$name" == "main" || "$name" == "followup" ]] && d="runs/$stem" || d="runs/${name}_$stem"
    [[ -f "$d/sweep.json" ]] && folders+=("$d")
  done
  if [[ ${#folders[@]} -gt 0 ]]; then
    "$PY" scripts/indaco/status.py "${folders[@]}"
  fi
  echo "== problems reported in the job logs (none is good news)"
  grep -H -E "preflight: |vLLM exited|vLLM did not answer|time limit near|run_all:|Traceback|Error:" \
    logs/dz-*.out 2> /dev/null | grep -v "smoke run fine" | tail -n 60 || true
} > "$summary" 2>&1
cat "$summary"

archive="results/disensa_${kind}_$stamp.tar.gz"
if [[ "$kind" == followup ]]; then
  tar -czf "$archive" "${folders[@]}" logs "$summary" 2> /dev/null
else
  tar -czf "$archive" runs/pilot_* runs/core_* runs/control_llama31_8b runs/replicate_llama31_8b \
    runs/_smoke logs "$summary" 2> /dev/null
fi
echo
echo "archive: $REPO/$archive ($(du -h "$archive" | cut -f1))"
echo "send it to Theo (it holds every run, every log and the summary above)"
