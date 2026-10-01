"""Command line: `disensa run <config>`, `disensa app`, `disensa ping <model>`."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer
from dotenv import load_dotenv

from disensa.config import ExperimentConfig
from disensa.runs import new_run_dir

app = typer.Typer(no_args_is_help=True, help="disensa: opinion dynamics with numeric or LLM agents")
load_dotenv()  # API keys from .env, if present


@app.command()
def run(config: Path, steps: int = typer.Option(None, help="override `steps`")):
    """Run an experiment from a YAML config (numeric or LLM, set by `dynamics.type`)."""
    from disensa.engine import run as run_experiment

    cfg = ExperimentConfig.from_yaml(config)
    if steps:
        cfg.steps = steps
    run_dir = new_run_dir(cfg)
    d = cfg.dynamics
    what = f"numeric/{d.rule}" if d.type == "numeric" else ", ".join(m.name for m in d.models)
    typer.echo(
        f"{cfg.name}: {cfg.network.n} agents on {cfg.network.topology}, {cfg.steps} steps, {what}"
    )
    t0 = time.time()

    def progress(t: int, m: dict) -> None:
        typer.echo(
            f"  step {t:3d}  pol {m['polarization']:.3f}  dis {m['global_disagreement']:.3f}  "
            f"nci {m['neighbor_correlation']:+.3f}  errors {m['errors']}  [{time.time() - t0:.0f}s]"
        )

    asyncio.run(run_experiment(cfg, run_dir, progress))
    typer.echo(f"saved to {run_dir}")


@app.command(name="app")
def app_cmd(
    port: int = typer.Option(8765, help="port on 127.0.0.1"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="open it in your browser"),
):
    """Open the web app: configure runs, start them, and watch them live."""
    from disensa.app.server import serve

    serve(port=port, open_browser=open_browser)


@app.command()
def ping(model: str, prompt: str = "In one sentence: is remote work good for society?"):
    """Send one prompt to a model, to check API keys or a local Ollama."""
    from disensa.llm import LLMClient

    resp = asyncio.run(
        LLMClient().complete(model, [{"role": "user", "content": prompt}], use_cache=False)
    )
    typer.echo(f"[{resp.model}] {resp.text}")


if __name__ == "__main__":
    app()
