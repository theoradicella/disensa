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
  run — log it in `error` (with `error_kind`) and keep the previous state. A belief must be a
  finite number; out of [-2, 2] it is clipped and flagged, keeping `belief_raw`. In the web
  pages, insert model text with `textContent` only.
- Every run reports how far it can be trusted and what it cost (`report.py`): per call in
  `events.jsonl`, per step in `metrics.parquet`, per run in `meta.json` `report`. LLM runs stop
  as `failed` when a step's failures exceed `max_error_rate`. `meta.json` `status` is
  running | finished | cancelled | failed.
- Sampling is explicit: temperature, top_p, top_k, `think` (off) and the sampling seed are sent
  on every call and are part of the cache key. vLLM models are `hosted_vllm/<model>` +
  `api_base`, never `openai/` (it refuses to send without an API key).
- The scorer (`scorer.py`) is a measurement only: it rates logged texts after a run and never
  feeds back. Self-rating is the simulation state; human validation (`validation.py`) decides
  which one the analysis uses.
- Old run folders (before 0.3) lack the new fields: everything that reads runs must still open
  them, showing "not recorded".
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
  engine.py          the simulation loop, run status and the failure guard
  metrics.py         polarization, global disagreement, neighbour correlation
  report.py          the run report: failures, tokens, time, out-of-range, granularity
  seeding.py         random streams: `seed` = the world, `replicate` = the dynamics
  runs.py            run folders: writing (RunWriter) and reading (load_run), config hash
  checks.py          warnings/errors on a config before it runs
  sweep.py           `disensa sweep`: grids of runs, resume, shards, sweep.json
  scorer.py          offline opinion scorer -> scores.parquet (measurement only)
  validation.py      human validation: sample export, Krippendorff's alpha, agreement report
  cli.py             `disensa run | sweep | score | validation | analyze | app | ping`
  dynamics/
    __init__.py      the Dynamics interface + make_dynamics
    numeric.py       DeGroot, bounded confidence, Friedkin–Johnsen
    llm_agent.py     LLM agents (Wang et al. §3), model assignment by weight
  llm/
    client.py        LiteLLM client, cache, JSON parsing/repair, per-provider thinking switch
    prompts.py       every prompt template (quoted verbatim in the thesis), scorer included
    serving.py       what served each model (Ollama digest/quantization, vLLM model ids)
    ollama.py        starting Ollama, installed models, downloads
  analysis/          many runs together (matplotlib only for figures: the `analysis` extra)
    load.py          find_run_dirs, load_runs -> one row per run
    summary.py       clusters, convergence, drift, movement, Wasserstein
    fit.py           numeric rules fitted to an LLM run on its own world, + baseline
    response.py      micro response curve and each rule's prediction
    figures.py       the thesis figures (PDF + CSV)
    analyze.py       `disensa analyze`
  app/
    server.py        stdlib HTTP server + JSON API, runs in threads
    payload.py       run folder -> JSON for the viewer
    web/             shared/, landing/ (/), lab/ (/lab), docs/ (/docs), vendor/d3.min.js
configs/             example configs; configs/sweeps/ example sweep files
scripts/indaco/      SLURM job-array template for vLLM on INDACO, and its README
```

## Commands

```bash
uv sync
uv run pytest && uv run ruff check . && uv run ruff format --check .
uv run disensa run configs/numeric_bc.yaml    # numeric, instant
uv run disensa run configs/quick.yaml         # Ollama, minutes
uv run disensa app                            # web app: configure, run, watch live
uv run disensa ping hosted_vllm/llama3.2:3b --api-base http://localhost:11434/v1
uv run disensa sweep configs/sweeps/numeric_bc_epsilon.yaml [--dry-run] [--shard i/N]
uv run disensa score runs/<run or sweep> --model ollama_chat/<scorer>
uv run disensa validation export runs/<sweep> --n 100 --out validation/<name>
uv run disensa validation report validation/<name>
uv run disensa analyze runs/<sweep> ... --out figures/
```

## Not done yet

- Running on INDACO: `scripts/indaco/sweep.slurm` is untested until access works; partitions,
  GPU memory and time limits are placeholders.
- The topic statements of `configs/sweeps/topics_core.yaml` (political, factual, trivial) are
  placeholders: Theo chooses them.
