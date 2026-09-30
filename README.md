# disensa

Opinion dynamics on social networks, run either with **classic numeric rules** (agent-based
models) or with **LLM agents** that read and write posts, possibly each on a different model.
It builds on the Social Simulation Framework of Wang et al. (COLING 2025, arXiv:2409.19338).

This is the minimal core. Features from the first prototype (personas, a scorer model, a
similarity recommender, rewiring, more numeric rules, analysis tools) are parked in the git
tag `v0.1-full` and will come back one at a time.

## Setup

```bash
uv sync
cp .env.example .env         # only if you use paid APIs
```

For free local models, install [Ollama](https://ollama.com):

```bash
ollama pull llama3.2:3b && ollama serve
uv run disensa ping ollama_chat/llama3.2:3b      # check it answers
```

## Run

```bash
uv run disensa app                               # the web app at http://127.0.0.1:8765
uv run disensa run configs/numeric_bc.yaml       # numeric, instant
uv run disensa run configs/demo_camps.yaml       # LLM pipeline with offline fake models, seconds
uv run disensa run configs/quick.yaml            # real LLM on Ollama, minutes
```

| Config | What it is |
|---|---|
| `numeric_bc.yaml` | Bounded confidence on a small world: opinion clusters |
| `numeric_degroot.yaml` | DeGroot on a scale-free network: consensus |
| `demo_fake.yaml`, `demo_camps.yaml` | LLM agents with offline fake models, free |
| `quick.yaml` | Smallest real run on Ollama |
| `mixed_local.yaml` | Three open-weight models from three labs, on Ollama |

`--steps N` overrides the number of steps for a quick try.

## How a run works

The loop is the same for both kinds of agents (`engine.py`):

```
G    <- build the network                 (Erdős–Rényi, Watts–Strogatz, Barabási–Albert)
x(0) <- draw initial opinions on [-2, 2]
for t = 1 .. steps:
    every agent looks at its neighbours at t-1
    x_i(t) <- update(i)                    numeric rule, or an LLM call
    record x(t) and the metrics
```

Updates are synchronous. The only thing that changes between experiments is `dynamics`:

```yaml
dynamics: {type: numeric, rule: bounded_confidence, epsilon: 0.7, mu: 0.5}
# or
dynamics:
  type: llm
  statement: Euthanasia should be legal for terminally ill adults who clearly request it.
  models: [{name: ollama_chat/llama3.2:3b}]
```

An LLM agent writes a first post at step 0. Each step it reads a few of its neighbours'
posts and replies (one call, JSON) with its reasoning, new belief, new post and updated
memory. Replies are cached on disk, so re-running a config costs nothing.

Metrics every step (Wang et al. §4.2): polarization, global disagreement, neighbour
correlation, mean opinion (per model when models are mixed).

## Code layout

```
src/disensa/
  config.py          the YAML schema
  network.py         graph + initial opinions
  engine.py          the loop
  metrics.py         polarization, disagreement, neighbour correlation
  runs.py            writing and reading run folders
  cli.py             `disensa run | app | ping`
  dynamics/          how an agent updates: numeric.py (DeGroot, bounded confidence), llm_agent.py
  llm/               client.py (LiteLLM + cache), fake.py (offline model), prompts.py
  app/               server.py (HTTP + JSON API), payload.py (run -> viewer JSON)
    web/             shared/ (base.css, header.js), landing/, lab/, docs/, vendor/d3.min.js
```

## Run outputs

Each run is a folder `runs/<timestamp>_<name>_s<seed>/`: `meta.json` (config, git commit),
`agents.parquet`, `edges.json`, `opinions.parquet`, `metrics.parquet`, and for LLM runs
`events.jsonl` (what each agent read, its reasoning, post, memory, prompt and raw reply).
See `runs.py`.

## Tests

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check .
```

Tests only use the offline `fake/` models; they never call real APIs.
