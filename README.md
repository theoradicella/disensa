# disensa

Opinion dynamics on networks, with **classic numeric models** (DeGroot, bounded confidence,
Friedkin–Johnsen) or **LLM agents** that exchange messages, each possibly on a different model.
Every agent's interactions, reasoning and messages are logged, so single trajectories can be
traced.

## Run the app

Install [uv](https://docs.astral.sh/uv/) once, then start the app:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --python 3.12 disensa app
```

It opens in your browser at http://127.0.0.1:8765. Choose **Numeric algorithms** for a run
that works straight away.

**LLM agents** need [Ollama](https://ollama.com) (free, runs models locally): install it, restart
the app, and download a model from the lab (e.g. Llama 3.2 3B, about 2 GB).

## From source

```bash
git clone https://github.com/theoradicella/disensa.git && cd disensa
uv sync
uv run disensa app                         # the web app
uv run disensa run configs/numeric_bc.yaml # a run from the command line
uv run pytest                              # tests (no model calls)
```

The app's **docs** page explains how a run works; run outputs are described in `src/disensa/runs.py`.
