# disensa — context for coding agents

Bachelor's thesis project (BAI, University of Milan; supervisors Flaminio Squazzoni, Stefano Zapperi).
LLM-agent opinion dynamics on social networks. It extends Wang et al. (COLING 2025,
arXiv:2409.19338), who used a single LLM for every agent, in two directions:

1. **Model heterogeneity**: different agents run on different LLMs (`ensemble.py`).
2. **Individual-level inspection**: every agent's reading, reasoning and posts are logged, so
   single trajectories can be traced and explained (`analysis.py`).

Research questions are deliberately not fixed yet — do not invent or add them.

## Conventions

- **Git: never add `Co-Authored-By: Claude`, `Claude-Session:` or any other Claude attribution to
  commit messages or PR descriptions, unless Theo explicitly asks.** Commits are authored by Theo only.
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
src/disensa/
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
  numeric.py    numeric baselines written in the run format (so they open in the app)
  export.py     run → JSON for the page; `viz` embeds it (and d3) into one offline file
  app.py        `disensa app`: local stdlib HTTP server + JSON API, runs in threads
  static/       landing.html (start page at /), docs.html (/docs), app.html (the lab at /lab: header that expands into
                the new-run form and runs list, plus the viewer), vendored d3.min.js. The project name is not shown in the UI for now;
                model text is untrusted: insert with textContent only
  cli.py        `disensa` command
```

## Commands

```bash
uv sync --extra analysis
uv run pytest && uv run ruff check . && uv run ruff format --check .
uv run disensa run configs/demo_fake.yaml     # offline, seconds
uv run disensa run configs/quick.yaml         # Ollama, minutes
uv run disensa inspect runs/<run_dir>         # per-model table + largest opinion shifts
uv run disensa inspect runs/<run_dir> --agent 7
uv run disensa app                            # web app: configure, run, watch live
uv run disensa viz runs/<run_dir>             # single offline viewer file for one run
```

## Not done yet

- M6: running on INDACO (vLLM + SLURM). The code only needs `api_base` pointed at a vLLM server.
