# Running the disensa experiments on INDACO

Everything happens on the INDACO login node, in three steps. The jobs check themselves,
wait for each other, requeue themselves at the time limit, and pack the results at the end:
nothing needs watching.

## 1. Install (once, ~1 hour, mostly downloading)

```bash
git clone disensa.bundle disensa        # or the repository link Theo sends
cd disensa
scripts/indaco/setup.sh
```

It installs everything and downloads the models (~90 GB, no account needed). It picks a
storage folder with enough space by itself and says which. Running it again is safe. It is
done when it prints `setup done. Next: scripts/indaco/run_all.sh`.

## 1b. Small test first (about an hour)

```bash
scripts/indaco/small_test.sh            # installs, downloads Llama 3.1 8B and the scorer, submits one job
scripts/indaco/small_test.sh check      # once the job has ended
```

One GPU job: Llama serves 7 runs of the real setup (`configs/experiments/small_test.yaml`:
4 with interactions, 3 without), then the scorer rates every message. The check prints each
run (failed replies, beliefs outside the scale, opinions at start and end, a message, the
scorer's agreement with the agents) and a list of `ok` / `FAIL` lines. Either way, send Theo
that output and `logs/dz-small_*.out`; he decides when the experiment starts. It does not
touch the experiment's runs.

## 2. Start the experiment, one topic type at a time

```bash
scripts/indaco/run_all.sh political     # first: the pilots, then the political statements
scripts/indaco/run_all.sh moral         # each next one once Theo has seen the previous
scripts/indaco/run_all.sh factual
scripts/indaco/run_all.sh trivial
```

Each finds the GPU partition by itself, submits its jobs and returns at once. A type is a
quarter of the work; with several GPUs free it takes a few hours. When its jobs have ended
(`run_all.sh status`), send Theo the archive from section 3. Start the next type only after
Theo says so. `scripts/indaco/run_all.sh` with no type runs everything that is left in one
go; never start it while a type's jobs are still running (it would start those runs twice).

## 3. Collect the results

At any time, to see where things stand:

```bash
scripts/indaco/run_all.sh status
```

Everything is finished when the job list at the top is empty and a file
`results/disensa_results_<date>.tar.gz` (a few GB) is listed at the bottom. Send Theo that
file, by OneDrive or WeTransfer. It holds every run, every log and a summary.

## If something went wrong

**The answer is almost always: run `scripts/indaco/run_all.sh` again.** It never redoes
finished work. It submits only what is missing and makes a new archive at the end. Some
cases:

| you see | what to do |
|---|---|
| `run_all: … is not downloaded` | `scripts/indaco/setup.sh` again (a download was interrupted) |
| `no folder with 120 GB free found` | ask INDACO where large files go, then put `SCRATCH_DIR="/that/folder"` in `scripts/indaco/cluster.local.env` and run setup again |
| `no partition with GPUs found` | put `PARTITION="<name>"` (from `sinfo -s`) in `scripts/indaco/cluster.local.env` |
| `sbatch: error: …` (account, QOS) | put `ACCOUNT="…"` or `QOS="…"` in `scripts/indaco/cluster.local.env` (INDACO's guide says which) |
| `run_all: files of the repository were changed` | `git status`, then `git checkout -- <file>`: the runs must match Theo's commit |
| the summary says a model's pilot `stopped at the failure guard` | nothing: that model is left out by design. Theo will see it in the summary |
| jobs pending for days | nothing to fix on our side; tell Theo if it is more than a week |
| anything else | send Theo the newest `results/summary_*.txt` and the `logs/` folder |

---

## What the jobs do (for reference)

`run_all.sh` submits a chain of SLURM jobs:

| job | what | waits for |
|---|---|---|
| `dz-check` | reports the GPU, memory and software of a node (`logs/dz-check_*.out`) | — |
| `dz-pilot_<model>` ×4 | 8 runs per model: is it reliable? | — |
| `dz-core_<model>` ×4 | 240 runs per model (3 GPUs each, as an array) | its pilot succeeding |
| `dz-control_llama31_8b` | 80 runs without interactions | the Llama pilot |
| `dz-replicate_llama31_8b` | 240 runs, second replicate | the Llama pilot |
| `dz-score-<sweep>` ×6 | the scorer rates every message of a sweep | that sweep |
| `dz-pack` | summary + archive in `results/` | everything above |

Every sweep job serves its model with vLLM on a free port of its node, then runs a 2-step
**smoke run**. If the model is not served, its replies are not valid JSON, or more than 5% of
them fail, the job stops there, within a minute, and the jobs waiting for it are cancelled.
Otherwise it runs its share of the sweep, 4 runs side by side. 15 minutes before the time
limit, a job puts itself back in the queue (up to 20 times). Finished runs are skipped when it
starts again.

Files: `cluster.env` (settings), `setup.sh`, `run_all.sh`, the job scripts (`sweep_job.sh`,
`score_job.sh`, `check_node.sh`, `pack_job.sh`, sharing `lib.sh`), `preflight.py` (the
checks) and `status.py` (the summary). The experiment itself is `configs/experiments/`:
never edit it, so that every run is the experiment described in the thesis. Each run
records the commit it was made with.

Never mix these runs with Ollama runs made on a Mac: the same model name is not the same
model (Ollama serves 4-bit weights, vLLM the full ones). Each run records what served it in
`meta.json` (`serving`).
