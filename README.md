# disensa

Opinion dynamics with LLM agents on social networks, where different agents can run on
**different language models**. It extends the Social Simulation Framework of Wang et al.
(COLING 2025, arXiv:2409.19338), which used one LLM for every agent, with:

- a **multi-model ensemble**: each agent's reasoning comes from a model you choose
  (Claude, GPT, Grok, DeepSeek, Llama, Gemma, Mistral, Qwen, …);
- **individual-level inspection**: every agent's reading, reasoning and posts are logged, so
  you can trace how and why a single agent moved;
- **network rewiring** (optional): agents unfollow people they disagree with;
- an **interactive d3 viewer** for every run.

## Setup

```bash
uv sync --extra analysis     # dependencies incl. analysis tools
cp .env.example .env         # only if you use paid APIs
```

For free local models, install [Ollama](https://ollama.com) and pull a few:

```bash
ollama pull llama3.2:3b && ollama pull llama3.1:8b && ollama pull qwen2.5:7b
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=3 ollama serve
uv run disensa ping ollama_chat/llama3.2:3b      # check it answers
```

## The app

```bash
uv run disensa app          # opens http://127.0.0.1:8765
```

The start page (a swarm of agents that scatters from your cursor) leads to the lab at `/lab`.

Pick a preset or set everything by hand (engine, topology, initial opinions, models,
recommender, rewiring, numeric algorithm), press **Run simulation**, and watch it live.
Nodes are shaded from pale blue (oppose) to deep blue (support) and drift left to right with
their opinion, so camps separate on screen. The **Runs** tab lists past runs to replay.
Start with the `demo_camps` preset: offline, a few seconds.

## Command line

```bash
uv run disensa run configs/demo_fake.yaml        # offline fake models, a few seconds
uv run disensa run configs/quick.yaml            # 10 agents, 5 days, Ollama, a few minutes
uv run disensa inspect runs/<run_dir>            # per-model behaviour + largest opinion shifts
uv run disensa inspect runs/<run_dir> --agent 3  # one agent's day-by-day story
uv run disensa viz runs/<run_dir>                # open the interactive viewer
```

| Config | What it is |
|---|---|
| `demo_camps.yaml` | Offline demo where the population splits into camps |
| `demo_fake.yaml`, `demo_fake_rewiring.yaml` | Offline demos (fake models); try the whole pipeline for free |
| `numeric_bc.yaml` | Classic bounded confidence, opens in the app like any run |
| `quick.yaml` | Smallest real run on Ollama, for checking prompts |
| `wang/{small_world,scale_free,random}.yaml` | Wang et al. replication, one model, 50 agents × 30 days |
| `mixed_local.yaml` | Three open-weight models from three labs, on Ollama |
| `mixed_api.yaml` | Claude, GPT, Grok, DeepSeek through their APIs (paid) |
| `bc_small_world.yaml`, `fj_scale_free.yaml` | Numeric baselines (`disensa baseline`) |

`--days N` overrides the number of days for a quick try.

## How a run works

1. **Network**: Erdős–Rényi, Watts–Strogatz or Barabási–Albert, seeded.
2. **Agents**: each gets a persona (gender, age, education, Big Five poles, initial
   opinion on [-2, 2]) and a model (`assignment`: balanced, random, or clustered by community).
3. **Day 0**: every agent writes a first post about the topic.
4. **Each day**: an agent reads a few posts from its neighbours (optionally only those
   within the recommender threshold), then returns its reasoning, new belief, new post and an
   updated memory in one call. A fixed scorer model rates every post on [-2, 2].
5. **End of day**: optional rewiring; metrics are recorded.

Everything is cached on disk, so re-running the same config costs nothing.

## Run outputs

`runs/<timestamp>_<name>_s<seed>/`

| File | Contents |
|---|---|
| `meta.json` | config, git commit, timing, call counts |
| `agents.parquet` | persona, model, degree per agent |
| `opinions.parquet` | per day and agent: self-reported belief, scorer rating, opinion used |
| `metrics.parquet` | per day: polarization, global disagreement, neighbour correlation, per-model means |
| `events.jsonl` | per day and agent: feed, reasoning, post, memory, raw reply, errors; rewiring events |
| `edges.jsonl` | the network on day 0 and on every day it changed |
| `viewer.html` | written by `disensa viz`: one offline file with the viewer and data |

## Layout

```
src/disensa/
  config.py     typed YAML experiment configs (Pydantic)
  networks.py   graphs and initial opinions
  baselines.py  DeGroot, bounded confidence, Deffuant, Friedkin–Johnsen
  metrics.py    polarization, global disagreement, neighbour correlation (Wang et al. §4.2)
  personas.py   Wang-style personas
  prompts.py    every prompt, in one place
  llm.py        one async client for all providers (LiteLLM) + cache + JSON repair
  fake_llm.py   deterministic offline backend
  engine.py     the simulation loop
  ensemble.py   model assignment
  rewiring.py   unfollow/follow dynamics
  analysis.py   timelines, opinion shifts, per-model summaries, change points
  numeric.py    numeric baselines in the run format
  export.py     run → viewer data / offline viewer file
  app.py        local web app server
  static/       the app page + vendored d3
  cli.py        `disensa` command
```

## Roadmap

- [x] M0 skeleton, configs, LLM client
- [x] M1 networks, numeric baselines, metrics
- [x] M2 LLM agents, Wang et al. loop, scorer
- [x] M3 multi-model ensemble
- [x] M4 rewiring
- [x] M5 individual trajectory tooling
- [x] M7 d3 viewer
- [x] Web app: configure, run and watch simulations live
- [ ] M6 INDACO: vLLM + SLURM sweeps (point `api_base` at a vLLM server)
