#!/bin/bash
# The whole thesis experiment on INDACO, in one command (on the login node):
#
#   scripts/indaco/run_all.sh           submit everything that is not done or queued yet
#   scripts/indaco/run_all.sh status    what is queued, running and done
#
# It submits a chain of SLURM jobs that wait for each other, then returns at once:
#
#   check                        what GPU the jobs get (information only)
#   pilot_<model>     ×4         8 runs each: is the model reliable?
#     core_<model>    ×4         240 runs each, only if its pilot passed (job exit 0)
#     control_<MAIN>             80 runs, only if the main model's pilot passed
#     replicate_<MAIN>           240 runs, idem
#   score_<sweep>                after each sweep above (not the pilots): the scorer rates
#                                every message
#   pack                         after everything: results/summary_*.txt and
#                                results/disensa_results_*.tar.gz, plus an e-mail if
#                                MAIL_USER is set
#
# A job near its time limit puts itself back in the queue. Running this script again is
# always safe: it skips what is finished or already queued and submits only the rest, so
# after any problem (a node failure, a job cancelled) the answer is: run it again.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck source=lib.sh
source scripts/indaco/lib.sh
# shellcheck source=cluster.env
source scripts/indaco/cluster.env
PY="$REPO/.venv/bin/python"
export LITELLM_LOCAL_MODEL_COST_MAP=True
mkdir -p logs

MODELS=(llama31_8b qwen25_7b mistral03_7b olmo2_7b)
MAIN=llama31_8b
E=configs/experiments

die() { echo "run_all: $*" >&2; exit 1; }
queued_id() { squeue -h -u "$USER" -n "$1" -o %A 2> /dev/null | head -n 1; }
state() { eval "$("$PY" scripts/indaco/preflight.py state "$1" --scorer "$SCORER")"; }

[[ -x "$PY" ]] || die "no disensa environment: run scripts/indaco/setup.sh first"

if [[ "${1:-}" == "status" ]]; then
  squeue -u "$USER" -o "%.14i %.32j %.9T %.11M %R" 2> /dev/null || true
  echo
  for f in "$E"/pilot/*.yaml "$E"/core/*.yaml "$E"/main/*.yaml; do
    "$PY" scripts/indaco/preflight.py state "$f" --scorer "$SCORER"
  done
  ls -1 results/*.tar.gz 2> /dev/null || true
  exit 0
fi
[[ $# -eq 0 ]] || die "usage: run_all.sh [status]"

# ---- before submitting anything: settings, weights, the repository ----------------------
command -v sbatch > /dev/null || die "no sbatch here: run this on the INDACO login node"
detect_slurm
for f in "$E"/pilot/*.yaml; do
  checked="$("$PY" scripts/indaco/preflight.py check "$f" 2> /dev/null)" || die "$f is not valid"
  eval "$checked"
  [[ -d "$HF_HOME/hub/models--${MODEL//\//--}" ]] || die "$MODEL is not downloaded: run scripts/indaco/setup.sh"
done
[[ -d "$HF_HOME/hub/models--${SCORER//\//--}" ]] || die "$SCORER is not downloaded: run scripts/indaco/setup.sh"
git diff --quiet -- src configs scripts/indaco ':!scripts/indaco/cluster.local.env' ||
  die "files of the repository were changed (git status): restore them, the runs must match the commit"

ALL=()                                   # every job id, for the final pack job
gpu=()
[[ -n "$GRES" ]] && gpu=(--gres="$GRES")
score_gpu=()
[[ -n "$SCORE_GRES" ]] && score_gpu=(--gres="$SCORE_GRES")

# submit_sweep <file> <tasks> <dependency or ""> : sets JOB (empty if nothing to do)
submit_sweep() {
  local file="$1" tasks="$2" dep="$3" id
  state "$file"
  id="$(queued_id "dz-$NAME")"
  if [[ -n "$id" ]]; then
    JOB="$id"; ALL+=("$id")
    echo "  $NAME: already queued or running (job $id)"
    return
  fi
  if ((FINISHED == TOTAL)); then
    JOB=""
    echo "  $NAME: done ($TOTAL runs)"
    return
  fi
  sbatch_common
  SB+=(${gpu[@]+"${gpu[@]}"} --time="$TIME" --requeue --signal=B:USR1@$REQUEUE_WARN --job-name="dz-$NAME")
  if ((tasks > 1)); then
    SB+=(--array="1-$tasks" --output="logs/%x_%A_%a.out")
  else
    SB+=(--output="logs/%x_%j.out")
  fi
  [[ -n "$dep" ]] && SB+=(--dependency="$dep" --kill-on-invalid-dep=yes)
  JOB="$(sbatch "${SB[@]}" scripts/indaco/sweep_job.sh "$file" | cut -d';' -f1)"
  ALL+=("$JOB")
  echo "  $NAME: submitted, job $JOB ($((TOTAL - FINISHED)) runs to do${dep:+, after ${dep#*:}})"
}

# submit_score <sweep file> <job id of that sweep or ""> : the scorer after the sweep
submit_score() {
  local file="$1" after="$2" id
  state "$file"
  id="$(queued_id "dz-score-$NAME")"
  if [[ -n "$id" ]]; then
    ALL+=("$id"); echo "  score_$NAME: already queued or running (job $id)"; return
  fi
  if [[ -z "$after" ]] && ((FINISHED == TOTAL && SCORED == FINISHED)); then
    echo "  score_$NAME: done"; return
  fi
  sbatch_common
  SB+=(${score_gpu[@]+"${score_gpu[@]}"} --time="$TIME" --requeue --signal=B:USR1@$REQUEUE_WARN
    --job-name="dz-score-$NAME" --output="logs/%x_%j.out")
  [[ -n "$after" ]] && SB+=(--dependency="afterany:$after")
  id="$(sbatch "${SB[@]}" scripts/indaco/score_job.sh "runs/$NAME" | cut -d';' -f1)"
  ALL+=("$id")
  echo "  score_$NAME: submitted, job $id${after:+, after $after}"
}

echo "disensa $(git rev-parse --short HEAD), $(date "+%Y-%m-%d %H:%M")"

if [[ -z "$(ls logs/dz-check_*.out 2> /dev/null)" && -z "$(queued_id dz-check)" ]]; then
  sbatch_common
  sbatch "${SB[@]}" ${gpu[@]+"${gpu[@]}"} --time=00:15:00 --job-name=dz-check \
    --output="logs/%x_%j.out" scripts/indaco/check_node.sh > /dev/null
  echo "  check: submitted (logs/dz-check_*.out: what GPU the jobs get)"
fi

for m in "${MODELS[@]}"; do
  echo "$m"
  pilot="$E/pilot/$m.yaml"
  state "$pilot"
  if [[ -z "$(queued_id "dz-$NAME")" ]] && ((FAILED > 0 && FINISHED + FAILED == TOTAL)); then
    echo "  $NAME: $FAILED pilot run(s) stopped at the failure guard: this model is left out"
    continue
  fi
  submit_sweep "$pilot" 1 ""
  dep="${JOB:+afterok:$JOB}"
  sweeps=("$E/core/$m.yaml")
  [[ "$m" == "$MAIN" ]] && sweeps+=("$E/main/control_$m.yaml" "$E/main/replicate_$m.yaml")
  for f in "${sweeps[@]}"; do
    tasks="$CORE_TASKS"
    [[ "$f" == *control_* ]] && tasks=1
    submit_sweep "$f" "$tasks" "$dep"
    submit_score "$f" "$JOB"
  done
done

old="$(queued_id dz-pack)"
[[ -n "$old" ]] && scancel "$old"        # replaced by one that waits for every job above
sbatch_common "$CPU_PARTITION"
SB+=(--time=02:00:00 --job-name=dz-pack --output="logs/%x_%j.out")
[[ ${#ALL[@]} -gt 0 ]] && SB+=(--dependency="afterany:$(IFS=:; echo "${ALL[*]}")")
[[ -n "$MAIL_USER" ]] && SB+=(--mail-type=END --mail-user="$MAIL_USER")
id="$(sbatch "${SB[@]}" scripts/indaco/pack_job.sh | cut -d';' -f1)"
echo "pack: submitted, job $id: when it ends, results/disensa_results_*.tar.gz is ready"
echo
echo "Nothing else to do now. Check with: scripts/indaco/run_all.sh status"
