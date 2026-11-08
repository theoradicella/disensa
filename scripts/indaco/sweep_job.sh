#!/bin/bash
# One sweep file on one GPU: serve its model with vLLM, check it with a 2-step smoke run,
# then run PARALLEL shards of the sweep side by side against the server.
# Submitted by run_all.sh; never run it by hand.
#
# As task t of a J-task array, the sweep is cut into J × PARALLEL shards and this task runs
# shards (t-1)·PARALLEL+1 … t·PARALLEL. Finished runs are always skipped, so starting the
# job again (after a requeue, or from run_all.sh) continues where it stopped.
#
# Exit status: 0 every run of this task finished; 1 the smoke run failed, or a run stopped.

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit with scripts/indaco/run_all.sh}"
SWEEP="${1:?usage: sweep_job.sh <sweep file>}"
cd "$REPO"
# shellcheck source=lib.sh
source "$REPO/scripts/indaco/lib.sh"
load_env
enable_requeue
log_env

checked="$("$PY" scripts/indaco/preflight.py check "$SWEEP")"
eval "$checked"                      # MODEL, SWEEP_NAME
start_vllm "$MODEL"

"$PY" scripts/indaco/preflight.py smoke "$SWEEP" \
  --api-base "$API_BASE" --out runs/_smoke \
  --cache-dir "$CACHE_ROOT/_smoke/$SWEEP_NAME" --max-failed "$MAX_SMOKE_FAILED"

TASK="${SLURM_ARRAY_TASK_ID:-1}"
TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
N=$((TASKS * PARALLEL))
pids=()
for j in $(seq 1 "$PARALLEL"); do
  i=$(((TASK - 1) * PARALLEL + j))
  log="logs/${JOBTAG}_shard${i}of${N}.log"
  echo "shard $i/$N -> $log"
  "$DISENSA" sweep "$SWEEP" --shard "$i/$N" --api-base "$API_BASE" \
    --cache-dir "$CACHE_ROOT/$SWEEP_NAME/shard${i}of${N}" >> "$log" 2>&1 &
  pids+=($!)
done

# wait for every shard; a signal (the time-limit warning) interrupts `wait`, so wait again
# until the shard has really ended, and keep its exit status
rc=0
for p in "${pids[@]}"; do
  while true; do
    code=0
    wait "$p" || code=$?
    kill -0 "$p" 2> /dev/null && continue
    [[ "$code" -ne 0 ]] && rc=1      # 1: a run stopped; the shard's other runs went on
    break
  done
done

"$PY" scripts/indaco/status.py "runs/$SWEEP_NAME" --parallel "$N"
exit "$rc"
