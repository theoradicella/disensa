#!/bin/bash
# A small test before the full experiment: one GPU job, about an hour.
#
#   scripts/indaco/small_test.sh          on the login node: install, download Llama 3.1 8B
#                                         (~16 GB) and the scorer (SCORER, ~30 GB), submit
#                                         the job
#   scripts/indaco/small_test.sh check    the result, once the job has ended
#
# The job serves Llama with vLLM and runs configs/experiments/small_test.yaml (7 runs: 4
# with interactions, 3 without), then serves the scorer and rates every message.
# Output: logs/dz-small_<job>.out, runs/small_test/. Running it again redoes the runs.

set -euo pipefail
# inside the job SLURM runs a copy of this file, so the repository is where it was submitted
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
SWEEP=configs/experiments/small_test.yaml

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # ---- inside the job --------------------------------------------------------------------
  # shellcheck source=lib.sh
  source scripts/indaco/lib.sh
  load_env
  log_env
  eval "$("$PY" scripts/indaco/preflight.py check "$SWEEP")"   # MODEL
  start_vllm "$MODEL"
  rm -rf runs/small_test
  "$DISENSA" sweep "$SWEEP" --api-base "$API_BASE" --cache-dir "$CACHE_ROOT/small_test" || true
  stop_vllm
  start_vllm "$SCORER" "$SCORE_VLLM_EXTRA_ARGS"
  echo "== scoring runs/small_test with $SCORER"
  "$DISENSA" score runs/small_test --model "hosted_vllm/$SCORER" --api-base "$API_BASE" \
    --field message --max-concurrency 128 --cache-dir "$CACHE_ROOT/score/small_test" || true
  "$PY" scripts/indaco/small_test.py runs/small_test
  exit 0
fi

if [[ "${1:-}" == "check" ]]; then
  .venv/bin/python scripts/indaco/small_test.py runs/small_test
  exit 0
fi

# ---- on the login node -------------------------------------------------------------------
# shellcheck source=cluster.env
source scripts/indaco/cluster.env
[[ -x .venv/bin/disensa && -x "$VLLM_ENV/bin/vllm" ]] || scripts/indaco/setup.sh env
scripts/indaco/setup.sh download "$SWEEP" "$SCORER"
source scripts/indaco/cluster.env      # setup.sh may have chosen SCRATCH_DIR
# shellcheck source=lib.sh
source scripts/indaco/lib.sh
detect_slurm
sbatch_common
mkdir -p logs
gres=()
[[ -n "$GRES" ]] && gres=(--gres="$GRES")
id="$(sbatch "${SB[@]}" ${gres[@]+"${gres[@]}"} --time=01:30:00 --job-name=dz-small \
  --output="logs/%x_%j.out" scripts/indaco/small_test.sh | cut -d';' -f1)"
echo "submitted job $id. Follow it with: tail -f logs/dz-small_$id.out"
echo "when it has ended: scripts/indaco/small_test.sh check"
