# agorasim — context for coding agents

Bachelor's thesis project (BAI, University of Milan; supervisors Flaminio Squazzoni, Stefano Zapperi).
LLM-agent opinion dynamics on social networks. It extends Wang et al. (COLING 2025,
arXiv:2409.19338), who used a single LLM for every agent, in two directions:

1. **Model heterogeneity**: different agents run on different LLMs (`ensemble.py`).
2. **Individual-level inspection**: every agent's reading, reasoning and posts are logged, so
   single trajectories can be traced and explained (`analysis.py`).

Research questions are deliberately not fixed yet — do not invent or add them.

## Conventions

- Opinions live on **[-2, 2]** everywhere (−2 strongly oppose, +2 strongly support the topic statement).
- Everything is seeded. Personas depend only on `seed` (never on models), so the same
  population can be replayed under different model mixes. Separate RNG streams:
  feeds `seed`, personas `seed+1000`, model assignment `seed+2000`, rewiring `seed+3000`.
- Days are synchronous: agents read day t−1 and all updates apply together.
- The model is an agent attribute, separate from the persona.
- One fixed scorer model rates every post (never the agent's own model) — Chuang et al.
- LLM calls go through `LLMClient` only (LiteLLM + disk cache). `fake/<behaviour>` models run
  offline and deterministically; use them in tests. Never call real APIs in tests.
- Model output is untrusted: parse with `parse_json`, and a malformed reply must never crash a
  run — log it in `error` and keep the previous state.
- Run outputs: see the docstring in `runs.py`. Parquet for tables, JSONL for text.
- Metrics follow Wang et al. §4.2 (`metrics.py`); cite the source in docstrings when adding more.

## Layout

```
src/agorasim/
  config.py     Pydantic schemas for YAML experiment configs
  networks.py   ER / WS / BA graphs, initial opinions
  baselines.py  DeGroot, bounded confidence, Deffuant, Friedkin–Johnsen
  metrics.py    polarization, global disagreement, neighbour correlation
  personas.py   Wang-style personas (gender, age, education, Big Five)
  prompts.py    every prompt template (quoted verbatim in the thesis)
  llm.py        LiteLLM client, cache, JSON parsing/repair
  fake_llm.py   offline deterministic backend
  engine.py     the simulation loop
  ensemble.py   model assignment strategies
  rewiring.py   unfollow/follow network dynamics
  analysis.py   loading runs, trajectories, shift detection and attribution
  export.py     run → JSON, embedded into viewer.html (the d3 viewer)
  viewer.html   single-file d3 viewer template (model text via textContent only)
  cli.py        `agorasim` command
```

## Commands

```bash
uv sync --extra analysis
uv run pytest && uv run ruff check . && uv run ruff format --check .
uv run agorasim run configs/demo_fake.yaml     # offline, seconds
uv run agorasim run configs/quick.yaml         # Ollama, minutes
uv run agorasim inspect runs/<run_dir>         # per-model table + largest opinion shifts
uv run agorasim inspect runs/<run_dir> --agent 7
uv run agorasim viz runs/<run_dir>             # open the d3 viewer
```

## Not done yet

- M6: running on INDACO (vLLM + SLURM). The code only needs `api_base` pointed at a vLLM server.
