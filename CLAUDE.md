# disensa — context for coding agents

Bachelor's thesis project (BAI, University of Milan; supervisors Flaminio Squazzoni, Stefano Zapperi).
LLM-agent opinion dynamics on social networks. It extends Wang et al. (COLING 2025,
arXiv:2409.19338), who used a single LLM for every agent, in two directions:

1. **Model heterogeneity**: different agents run on different LLMs.
2. **Individual-level inspection**: every agent's interactions, reasoning and messages are logged, so
   single trajectories can be traced and explained.

Research questions are deliberately not fixed yet — do not invent or add them.

## Conventions

- **Git: never add `Co-Authored-By: Claude`, `Claude-Session:` or any other Claude attribution to
  commit messages or PR descriptions, unless Theo explicitly asks.** Commits are authored by Theo only.
- **Minimal core first.** The first prototype is parked in the git tag `v0.1-full` (personas,
  scorer, recommender, rewiring, Deffuant/Friedkin–Johnsen, analysis, offline viewer). Bring
  features back one at a time, only when Theo asks.
- One loop for everything (`engine.py`). Numeric and LLM agents differ only in the `Dynamics`
  object (`dynamics/`): `labels()`, `start(x0)`, `step(x, t) -> (new x, events)`.
- Opinions live on **[-2, 2]** everywhere (−2 strongly oppose, +2 strongly support the statement).
- Everything is seeded (`seeding.py`). `seed` = the world (network, opinions, model assignment),
  the same for numeric and LLM runs; `replicate` = the dynamics (interactions, LLM sampling
  seed). Every stream has its own `SeedSequence`; never reuse one stream for two draws.
- Steps are synchronous: agents use the state of step t−1 and all updates apply together.
- **Vocabulary: no social-media framing.** LLM agents have *interactions*; each interaction is
  an *author* and the *message* they shared at t−1. Say *step*, never *day*; *message*, never
  *post*; *interactions*, never *feed*. Prompts describe a community, not a platform. Same
  mechanism as Wang et al.
- LLM calls go through `LLMClient` only (LiteLLM + disk cache). The app has no fake models.
  Tests replace `LLMClient._call_litellm` with the offline stand-in in `tests/fake_llm.py`
  (`conftest.py`, autouse); never call real APIs in tests.
- Model output is untrusted: parse with `parse_json`, and a malformed reply must never crash a
  run — log it in `error` and keep the previous state. In the web pages, insert model text
  with `textContent` only.
- Run outputs: see the docstring in `runs.py`. Parquet for tables, JSONL for text.
- Metrics follow Wang et al. §4.2 (`metrics.py`); cite the source in docstrings when adding more.
- Frontend: plain HTML/CSS/JS, no build step. One folder per page under `app/web/`
  (`index.html` + its own `.css`/`.js`); the header is `<site-header>` from `shared/header.js`,
  shared styles in `shared/base.css`. The project name is not shown in the UI for now.

## Layout

```
src/disensa/
  config.py          Pydantic schema for YAML configs (`dynamics.type`: numeric | llm)
  network.py         ER / WS / BA graphs, initial opinions
  engine.py          the simulation loop
  metrics.py         polarization, global disagreement, neighbour correlation
  runs.py            run folders: writing (RunWriter) and reading (load_run)
  cli.py             `disensa run | app | ping`
  dynamics/
    __init__.py      the Dynamics interface + make_dynamics
    numeric.py       DeGroot, bounded confidence
    llm_agent.py     LLM agents (Wang et al. §3), model assignment by weight
  llm/
    client.py        LiteLLM client, cache, JSON parsing/repair
    prompts.py       every prompt template (quoted verbatim in the thesis)
  app/
    server.py        stdlib HTTP server + JSON API, runs in threads
    payload.py       run folder -> JSON for the viewer
    web/             shared/, landing/ (/), lab/ (/lab), docs/ (/docs), vendor/d3.min.js
```

## Commands

```bash
uv sync
uv run pytest && uv run ruff check . && uv run ruff format --check .
uv run disensa run configs/numeric_bc.yaml    # numeric, instant
uv run disensa run configs/quick.yaml         # Ollama, minutes
uv run disensa app                            # web app: configure, run, watch live
```

## Not done yet

- `docs/index.html` still describes the full prototype; to be rewritten.
- Running on INDACO (vLLM + SLURM). The code only needs `api_base` pointed at a vLLM server.
