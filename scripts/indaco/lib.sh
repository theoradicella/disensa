# Shared by the scripts of scripts/indaco/ (sourced, not run).
# Expects REPO (the repository root) to be set.

load_env() {
  # shellcheck source=/dev/null
  source "$REPO/scripts/indaco/cluster.env"
  if [[ -n "${MODULES:-}" ]]; then
    # shellcheck disable=SC2086
    module load $MODULES
  fi
  export HF_HOME
  export HF_HUB_OFFLINE=1           # jobs never download: setup.sh did that on the login node
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  export LITELLM_LOCAL_MODEL_COST_MAP=True   # LiteLLM would download its price list at start
  PY="$REPO/.venv/bin/python"
  DISENSA="$REPO/.venv/bin/disensa"
  [[ -x "$DISENSA" ]] || { echo "no disensa environment: run scripts/indaco/setup.sh" >&2; exit 1; }
  JOBTAG="${SLURM_JOB_NAME:-job}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0}"
  mkdir -p "$REPO/logs" "$CACHE_ROOT"
}

log_env() {
  echo "== $(date "+%Y-%m-%d %H:%M:%S") on $(hostname), job ${SLURM_JOB_ID:-?}," \
    "restart ${SLURM_RESTART_COUNT:-0}, GPUs ${CUDA_VISIBLE_DEVICES:-?}"
  echo "disensa commit $(git -C "$REPO" rev-parse --short HEAD 2> /dev/null || echo '?')" \
    "$(git -C "$REPO" diff --quiet 2> /dev/null || echo '(with uncommitted changes!)')"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2> /dev/null || true
  "$VLLM_ENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)" 2> /dev/null || true
}

# sbatch options shared by every job: partition, account, QOS. Add --gres yourself.
sbatch_common() {
  local p="${1:-$PARTITION}"
  SB=(--partition="$p" --cpus-per-task="$CPUS" --mem="$MEM" --parsable --open-mode=append)
  [[ -n "$ACCOUNT" ]] && SB+=(--account="$ACCOUNT")
  [[ -n "$QOS" ]] && SB+=(--qos="$QOS")
  return 0
}

# A job submitted with `--requeue --signal=B:USR1@<REQUEUE_WARN>` gets USR1 that many seconds
# before its time limit and puts itself back in the queue. Finished runs are skipped when it starts again
# and the caches replay the unfinished ones, so nothing is lost but the queue wait.
enable_requeue() {
  trap on_time_limit USR1
}

on_time_limit() {
  local id="${SLURM_JOB_ID:-}"
  [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] && id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  if ((${SLURM_RESTART_COUNT:-0} < MAX_REQUEUE)); then
    echo "== $(date "+%H:%M:%S") time limit near: back in the queue ($id), it will continue"
    scontrol requeue "$id"
  else
    echo "== time limit near, already requeued $MAX_REQUEUE times: stopping." \
      "Run scripts/indaco/run_all.sh again to continue."
  fi
}

# detect_slurm: fill PARTITION and TIME when cluster.local.env leaves them empty.
#   PARTITION  the partition whose nodes have GPUs (sinfo's GRES column mentions "gpu");
#              with several, the one with the longest time limit
#   TIME       that partition's time limit, or 24 h if it has none
# Prints what it chose. Fails only when no partition has GPUs.
detect_slurm() {
  if [[ -z "$PARTITION" ]]; then
    local best="" best_min=-1 p lim min
    while read -r p lim; do
      min="$(slurm_minutes "$lim")"
      if ((min > best_min)); then best="$p" best_min="$min"; fi
    done < <(sinfo -h -o "%P %l %G" 2> /dev/null | awk '$3 ~ /gpu/ {gsub(/\*/, "", $1); print $1, $2}' | sort -u)
    [[ -n "$best" ]] || {
      echo "no partition with GPUs found by sinfo: set PARTITION in scripts/indaco/cluster.local.env" >&2
      exit 1
    }
    PARTITION="$best"
    CPU_PARTITION="${CPU_PARTITION:-$PARTITION}"
  fi
  if [[ -z "$TIME" ]]; then
    TIME="$(sinfo -h -p "$PARTITION" -o %l 2> /dev/null | head -n 1)"
    [[ -z "$TIME" || "$TIME" == "infinite" || "$TIME" == "UNLIMITED" ]] && TIME="24:00:00"
  fi
  echo "SLURM: partition $PARTITION, time limit $TIME per job"
}

# SLURM time ("2-00:00:00", "12:00:00", "30:00", "infinite") -> minutes
slurm_minutes() {
  local t="$1" d=0 h=0 m=0
  [[ "$t" == "infinite" || "$t" == "UNLIMITED" ]] && { echo 999999; return; }
  [[ "$t" == *-* ]] && { d="${t%%-*}"; t="${t#*-}"; }
  IFS=: read -r -a parts <<< "$t"
  case "${#parts[@]}" in
    3) h="${parts[0]}" m="${parts[1]}" ;;
    2) m="${parts[0]}" ;;
    1) m="${parts[0]}" ;;
  esac
  echo $(((10#$d * 24 + 10#$h) * 60 + 10#$m))
}

free_port() {
  "$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# start_vllm <Hugging Face id> [extra vLLM arguments]: serve it on a free local port; sets
# VLLM_PID and API_BASE. The port is chosen per job, so two jobs on the same node never talk
# to each other's server.
start_vllm() {
  local model="$1" extra="${2:-$VLLM_EXTRA_ARGS}"
  [[ -x "$VLLM_ENV/bin/vllm" ]] || { echo "no vLLM in $VLLM_ENV: run scripts/indaco/setup.sh" >&2; exit 1; }
  local port
  port="$(free_port)"
  API_BASE="http://127.0.0.1:$port/v1"
  VLLM_LOG="$REPO/logs/vllm_${JOBTAG}.log"
  echo "starting vLLM: $model on port $port (log: $VLLM_LOG)"
  # --served-model-name: the name disensa sends (hosted_vllm/<this>)
  # --generation-config vllm: ignore the model's own default sampling settings (e.g. a
  #   repetition penalty); disensa sends temperature, top_p and top_k on every request
  # --seed 0: with disensa's per-request seed, sampling is seeded (GPU batching can still
  #   change results slightly; the reply cache keeps what was actually generated)
  # shellcheck disable=SC2086
  "$VLLM_ENV/bin/vllm" serve "$model" \
    --served-model-name "$model" \
    --host 127.0.0.1 --port "$port" \
    --dtype "$VLLM_DTYPE" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEM" \
    --generation-config vllm \
    --seed 0 \
    $extra >> "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  trap stop_vllm EXIT
  local tries=$((VLLM_WAIT_MIN * 12)) waited=0
  for _ in $(seq 1 "$tries"); do
    if curl -sf "$API_BASE/models" > /dev/null 2>&1; then
      echo "vLLM is up after $waited s"
      return 0
    fi
    if ! kill -0 "$VLLM_PID" 2> /dev/null; then
      echo "vLLM exited while starting. Last lines of $VLLM_LOG:" >&2
      tail -n 30 "$VLLM_LOG" >&2
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "vLLM did not answer within $VLLM_WAIT_MIN min. Last lines of $VLLM_LOG:" >&2
  tail -n 30 "$VLLM_LOG" >&2
  exit 1
}

stop_vllm() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2> /dev/null || true
    wait "$VLLM_PID" 2> /dev/null || true
  fi
}
