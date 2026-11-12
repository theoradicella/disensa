# The thesis experiments

Every LLM sweep here runs on INDACO with vLLM (`scripts/indaco/README.md`). The numeric
baselines run on the Mac. Do not edit a file after its runs have started: a run is
recognised by its config hash, so a changed file makes new runs.

| file | what | runs |
|---|---|---:|
| `topics.csv` | the 16 statements: 4 types × 2 topics × 2 framings (`a`, `b`); `true` marks the true framing of a factual topic | |
| `base.yaml` | what every LLM sweep shares: 50 agents, uniform starting opinions, 30 steps, 1–5 interactions per step; a run stops only if more than half of a step's replies fail | |
| `pilot/<model>.yaml` | each candidate model on 8 statements (framing `a`), one world (seed 100), Watts–Strogatz | 8 each |
| `core/<model>.yaml` | 5 worlds (seeds 1–5) × 3 topologies × 16 statements | 240 each |
| `main/control_<model>.yaml` | main model, no interactions (`interactions.max: 0`), 5 worlds × 16 statements | 80 |
| `main/replicate_<model>.yaml` | main model, the core grid again with replicate 1 | 240 |
| `followup/control_v2_<model>.yaml` | every model, no interactions, as `main/control_*` but with a new sampling seed at every step (`dynamics.seed_per_step`): with one seed per run, an agent that hears nobody gets the same prompt again and so the very same reply, and every agent of the first control was frozen from about step 5 | 80 each |
| `followup/replicate_<model>.yaml` | the three other models, a small second replicate: world 1 × 3 topologies × 16 statements, replicate 1, so each model has its own replicate floor | 48 each |
| `numeric_baselines.yaml` | DeGroot, bounded confidence, Friedkin–Johnsen on the core worlds (Mac) | 45 |

Topologies, all with 50 agents and mean degree ≈ 6: Watts–Strogatz (k = 6, β = 0.1),
Barabási–Albert (m = 3), Erdős–Rényi (p = 0.12). A seed fixes the network *and* the
starting opinions, so seed s gives every statement, topology and model the same starting
opinions: all comparisons are paired.

Models (vLLM, 16-bit, temperature 0.7, top_p 0.9, top_k 40, thinking off):

| short name | Hugging Face id | lab |
|---|---|---|
| `llama31_8b` | `unsloth/Llama-3.1-8B-Instruct` | Meta |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` | Alibaba |
| `mistral03_7b` | `mistralai/Mistral-7B-Instruct-v0.3` | Mistral AI |
| `olmo2_7b` | `allenai/OLMo-2-1124-7B-Instruct` | AI2 |

Llama comes from `unsloth/`, a public copy of Meta's `meta-llama/Llama-3.1-8B-Instruct`
that needs no Hugging Face account: its four weight files have the same SHA-256 checksums as
Meta's (checked 2026-11-08). Its licence is still Meta's Llama 3.1 Community License.

Every model whose pilot passes (its smoke run and all 8 pilot runs finish) runs the core
grid; the pilot judges reliability only, never how the opinions moved. `main/` (control
and second replicate) is for `llama31_8b`, the main model. `scripts/indaco/run_all.sh`
submits all of it as one chain of jobs, the scorer included. The follow-up sweeps were added
after the experiment had run (2026-11-11) and go with `run_all.sh followup`; they use the
experiment's worlds, so every follow-up run is paired with core runs. For the analysis,
`control_v2_llama31_8b` replaces `main/control_llama31_8b`, which is kept only to show why.

`tests/test_experiments.py` checks that every sweep uses only these statements, one vLLM
model with these sampling settings, and the same worlds.

Analysis, on the Mac, one model at a time (`analyze` groups runs by statement, so mixing
models in one call would pool them):

```bash
uv run disensa sweep configs/experiments/numeric_baselines.yaml
uv run --extra analysis disensa analyze runs/core_<model> runs/numeric_baselines --out figures/<model>
```
