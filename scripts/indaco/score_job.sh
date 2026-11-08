#!/bin/bash
# Rate every message of the finished runs of some sweeps with the scorer (SCORER in
# cluster.env): a measurement after the runs, it never changes them. Writes scores.parquet
# in each run folder. Submitted by run_all.sh after the sweeps it scores.
#   score_job.sh runs/<sweep> [runs/<sweep> ...]
# Runs already scored, and runs not finished, are skipped, so it can always start again.
# FIELD=reasoning in the environment rates the reasoning instead of the message.

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit with scripts/indaco/run_all.sh}"
cd "$REPO"
# shellcheck source=lib.sh
source "$REPO/scripts/indaco/lib.sh"
load_env
enable_requeue
log_env

folders=()
for f in "$@"; do
  if [[ -f "$f/sweep.json" ]]; then folders+=("$f"); else echo "$f: no runs, nothing to score"; fi
done
[[ ${#folders[@]} -gt 0 ]] || exit 0

start_vllm "$SCORER" "$SCORE_VLLM_EXTRA_ARGS"
for f in "${folders[@]}"; do
  echo "== scoring $f (${FIELD:-message}) with $SCORER"
  "$DISENSA" score "$f" --model "hosted_vllm/$SCORER" --api-base "$API_BASE" \
    --field "${FIELD:-message}" --max-concurrency 128 --resume \
    --cache-dir "$CACHE_ROOT/score/$(basename "$f")" &
  p=$!
  while true; do               # in the background, so the time-limit signal is handled at once
    code=0
    wait "$p" || code=$?
    kill -0 "$p" 2> /dev/null || break
  done
  [[ "$code" -eq 0 ]] || { echo "scoring $f ended with status $code"; rc=1; }
done
echo "== scoring done"
exit "${rc:-0}"
