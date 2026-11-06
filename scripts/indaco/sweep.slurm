#!/bin/bash
# disensa sweep as a SLURM job array on INDACO (University of Milan HPC).
#
# Each array task starts its own vLLM server on its GPU, runs its share of the sweep
# against it, and stops it. Submit from the repository root:
#
#   sbatch --array=1-4 scripts/indaco/sweep.slurm configs/sweeps/topics_core.yaml
#
# The array must be 1-based (1-N): task i runs shard i/N. Every value marked
# CHECK WITH INDACO is a placeholder: partitions, GPU types, memory and time limits are
# not confirmed yet (see scripts/indaco/README.md).
#
# UNTESTED until INDACO access works.

#SBATCH --job-name=disensa-sweep
#SBATCH --partition=CHECK_WITH_INDACO        # the GPU partition
#SBATCH --gres=gpu:1                         # CHECK WITH INDACO: GPU type/count syntax
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                            # CHECK WITH INDACO
#SBATCH --time=12:00:00                      # CHECK WITH INDACO: the partition's limit
#SBATCH --output=logs/%x_%A_%a.out           # mkdir -p logs before submitting

set -euo pipefail

SWEEP_FILE="${1:?usage: sbatch --array=1-N scripts/indaco/sweep.slurm <sweep file>}"
# the Hugging Face model vLLM serves; disensa configs must name it hosted_vllm/$MODEL
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
SCRATCH="${SCRATCH:-$HOME/scratch}"         # CHECK WITH INDACO: the scratch filesystem

# 1. modules and environment ------------------------------------------------------------
# CHECK WITH INDACO: module names. One Python env with both disensa and vLLM, e.g.
#   uv venv ~/envs/disensa && uv pip install --python ~/envs/disensa disensa vllm
# module load cuda/12.1
source "$HOME/envs/disensa/bin/activate"
export HF_HOME="$SCRATCH/hf"                # model weights are large: keep them on scratch

# 2. start vLLM in the background --------------------------------------------------------
vllm serve "$MODEL" --port "$PORT" --seed 0 > "logs/vllm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

# 3. wait until it answers GET /v1/models (loading weights can take minutes) -------------
for i in $(seq 1 180); do                   # 180 × 5 s = 15 min
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM exited, see its log" >&2; exit 1; fi
  sleep 5
done
curl -sf "http://localhost:$PORT/v1/models" > /dev/null || { echo "vLLM did not start" >&2; exit 1; }
echo "vLLM is serving $MODEL on port $PORT"

# 4. this task's share of the sweep -------------------------------------------------------
# Each task gets its own cache folder: the cache is SQLite (diskcache), which is not safe to
# share between processes over a network filesystem.
disensa sweep "$SWEEP_FILE" \
  --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT" \
  --cache-dir "$SCRATCH/disensa-cache/$SLURM_ARRAY_TASK_ID"

# 5. vLLM is stopped by the trap on exit -------------------------------------------------
