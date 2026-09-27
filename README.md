# agorasim

Opinion dynamics with LLM agents on social networks, where different agents run on
**different language models**. Extends Wang et al. (COLING 2025) with a multi-model
ensemble and individual-level trajectory inspection.

## Setup

```bash
uv sync                      # core + dev tools
uv sync --extra analysis     # + DuckDB, statsmodels, ruptures, plotting
cp .env.example .env         # add API keys you need
```

## Usage

```bash
uv run agorasim baseline configs/bc_small_world.yaml   # numeric baseline + metrics
uv run agorasim ping ollama/llama3.1                   # check a model is reachable
uv run pytest
```

Runs are written to `runs/<timestamp>_<name>_s<seed>/`:
`meta.json` (config + git commit), `trajectory.parquet` (step, agent, opinion),
`metrics.parquet`.

## Layout

```
src/agorasim/
  config.py     typed YAML experiment configs (Pydantic)
  networks.py   Erdős–Rényi / Watts–Strogatz / Barabási–Albert, initial opinions
  baselines.py  DeGroot, bounded confidence, Deffuant, Friedkin–Johnsen
  metrics.py    polarization, global disagreement, neighbour correlation (Wang et al. §4.2)
  llm.py        one async client for every provider (LiteLLM) + disk cache + fake backend
  runs.py       run directories and Parquet output
  cli.py        `agorasim` command
```

## Models

Any LiteLLM model string works: `anthropic/…`, `openai/…`, `xai/…`, `deepseek/…`,
`ollama/…` (local), or `openai/<model>` with `api_base` pointing at a vLLM server (INDACO).
`fake/<anything>` is a deterministic offline backend for tests.

## Roadmap

- [x] M0 — skeleton, configs, LLM client, caching
- [x] M1 — networks, numeric baselines, metrics
- [ ] M2 — single-model LLM replication of Wang et al. (personas, memory, classifier scoring)
- [ ] M3 — multi-model ensemble
- [ ] M4 — rewiring + recommender
- [ ] M5 — individual trajectory tooling
- [ ] M6 — INDACO: vLLM + SLURM sweeps
- [ ] M7 — d3 visualization
