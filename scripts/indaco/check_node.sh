#!/bin/bash
# What a GPU node offers, and whether vLLM can run there. Submit with
# scripts/indaco/run_all.sh (the dz-check job; its log is logs/dz-check_<job>.out).
# Runs no model, downloads nothing; takes a minute.

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit with scripts/indaco/run_all.sh}"
cd "$REPO"
# shellcheck source=/dev/null
source "$REPO/scripts/indaco/cluster.env"
if [[ -n "${MODULES:-}" ]]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

echo "== node: $(hostname), $(date "+%Y-%m-%d %H:%M:%S"), partition ${SLURM_JOB_PARTITION:-?}"
echo "CPUs: $(nproc), memory: $(free -g 2> /dev/null | awk '/Mem:/ {print $2 " GB"}')"
echo "== GPUs"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2> /dev/null ||
  nvidia-smi 2> /dev/null || echo "nvidia-smi: not found (no GPU on this node?)"
echo "== disk"
df -h "$HOME" "$SCRATCH_DIR" 2> /dev/null
echo "== internet (Hugging Face) from the GPU node"
if curl -sI --max-time 8 https://huggingface.co > /dev/null; then
  echo "reachable"
else
  echo "NOT reachable: weights must be downloaded on the login node (setup.sh download)"
fi
echo "== software"
"$REPO/.venv/bin/disensa" --help > /dev/null 2>&1 && echo "disensa: ok" || echo "disensa: missing (setup.sh env)"
if [[ -x "$VLLM_ENV/bin/python" ]]; then
  "$VLLM_ENV/bin/python" - << 'EOF'
import sys
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:  # noqa: BLE001
    print("vllm: cannot import:", e)
try:
    import torch
except ImportError:
    sys.exit("torch: not installed in the vLLM environment (run scripts/indaco/setup.sh)")
print("torch", torch.__version__, "cuda", torch.version.cuda, "| GPU visible:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    gb = torch.cuda.get_device_properties(i).total_memory / 2**30
    print(f"GPU {i}: {name}, compute capability {cap[0]}.{cap[1]}, {gb:.0f} GB")
    if cap < (7, 0):
        print("  -> vLLM does not support this GPU. Tell Theo (fallback: Ollama on the node).")
    elif cap < (8, 0):
        print('  -> no bfloat16: set VLLM_DTYPE="half" in cluster.env. Recent vLLM versions may')
        print("     not support this GPU at all: try the pilot and send its log if it fails.")
    else:
        print("  -> fine for vLLM, bfloat16 (VLLM_DTYPE=auto).")
    if gb < 22:
        print("  -> too small for the 7-8B models in 16 bits (~16 GB of weights + cache).")
    elif gb < 38:
        print("  -> fine for the 7-8B agent models; the 14B scorer needs a 40 GB+ GPU or 2 GPUs.")
    else:
        print("  -> fine for the agent models and the 14B scorer.")
EOF
else
  echo "vLLM environment missing ($VLLM_ENV): run scripts/indaco/setup.sh env"
fi
