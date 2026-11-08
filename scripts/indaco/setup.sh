#!/bin/bash
# One-time setup on the INDACO login node (it has internet; GPU nodes may not):
#
#   scripts/indaco/setup.sh         everything: uv, the disensa env, the vLLM env, and the
#                                   weights of the 4 agent models and the scorer (~90 GB)
#
# Safe to run again: what is already installed or downloaded is kept. The parts alone:
#   scripts/indaco/setup.sh env
#   scripts/indaco/setup.sh download <sweep file or Hugging Face id>...

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck source=cluster.env
source scripts/indaco/cluster.env
mkdir -p logs

setup_env() {
  if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  # disensa, exactly as locked in uv.lock (its own Python, no module needed)
  uv sync --frozen --python 3.12
  # vLLM in a separate environment: it pins its own torch and CUDA libraries
  [[ -x "$VLLM_ENV/bin/python" ]] || uv venv --python 3.12 "$VLLM_ENV"
  uv pip install --python "$VLLM_ENV/bin/python" "vllm${VLLM_VERSION:+==$VLLM_VERSION}"
  {
    echo "setup $(date "+%Y-%m-%d %H:%M:%S") on $(hostname)"
    echo "disensa $(.venv/bin/python -c 'import disensa; print(disensa.__version__)')"
    "$VLLM_ENV/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)"
  } | tee logs/setup_env.txt
}

download() {
  local models=() a m
  for a in "$@"; do
    if [[ -f "$a" ]]; then
      while read -r m; do models+=("$m"); done < <(.venv/bin/python scripts/indaco/preflight.py models "$a")
    else
      models+=("$a")
    fi
  done
  export HF_HOME
  for m in $(printf '%s\n' "${models[@]}" | sort -u); do
    echo "== $m -> $HF_HOME"
    # skip the duplicate original-format weights some repositories also ship
    "$VLLM_ENV/bin/python" - "$m" << 'EOF'
import sys
from huggingface_hub import snapshot_download
try:
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
except ImportError:  # older huggingface_hub
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

repo = sys.argv[1]
try:
    path = snapshot_download(repo, ignore_patterns=["original/*", "consolidated*", "*.pth", "*.gguf"])
except GatedRepoError:
    sys.exit(f"{repo} is gated: accept its licence on huggingface.co/{repo} with the account "
             "of your token, and export HF_TOKEN=hf_... before running this again")
except RepositoryNotFoundError:
    sys.exit(f"{repo}: no such model on Hugging Face (or the token cannot see it)")
print("ok:", path)
EOF
  done
  du -sh "$HF_HOME/hub" 2> /dev/null || true
}

# choose_storage: the weights (~90 GB) need a folder with room. Unless SCRATCH_DIR is set in
# cluster.local.env, try $SCRATCH, /scratch/$USER, /scratch/users/$USER and $HOME/scratch, take the first one with 120 GB free, and remember it in cluster.local.env so
# the jobs use the same folder.
choose_storage() {
  local local_env=scripts/indaco/cluster.local.env need=$((120 * 1024 * 1024)) d free
  if [[ -f "$local_env" ]] && grep -q '^SCRATCH_DIR=' "$local_env"; then
    mkdir -p "$SCRATCH_DIR"
    echo "storage: $SCRATCH_DIR (from cluster.local.env)"
    return
  fi
  for d in ${SCRATCH:+"$SCRATCH"} "/scratch/$USER" "/scratch/users/$USER" "$HOME/scratch"; do
    mkdir -p "$d" 2> /dev/null || continue
    [[ -w "$d" ]] || continue
    free="$(df -Pk "$d" | awk 'NR == 2 {print $4}')"
    # already downloaded counts as room: running setup again must not fail
    if ((free >= need)) || [[ -d "$d/hf/hub" ]]; then
      SCRATCH_DIR="$d"
      echo "SCRATCH_DIR=\"$d\"" >> "$local_env"
      # shellcheck source=cluster.env
      source scripts/indaco/cluster.env   # recompute HF_HOME and CACHE_ROOT from it
      echo "storage: $SCRATCH_DIR ($((free / 1024 / 1024)) GB free), saved in $local_env"
      return
    fi
    echo "storage: $d has only $((free / 1024 / 1024)) GB free"
  done
  echo "no folder with 120 GB free found: write SCRATCH_DIR=\"<a large folder>\" in $local_env" >&2
  exit 1
}

case "${1:-all}" in
  all)
    choose_storage
    setup_env
    download configs/experiments/pilot/*.yaml "$SCORER"
    echo
    echo "setup done. Next: scripts/indaco/run_all.sh"
    ;;
  env) setup_env ;;
  download)
    shift
    choose_storage
    [[ $# -gt 0 ]] || { echo "usage: setup.sh download <sweep file or Hugging Face id>..." >&2; exit 1; }
    download "$@"
    ;;
  *)
    sed -n 2,12p "$0"
    exit 1
    ;;
esac
