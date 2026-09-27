"""`agorasim app`: a local web app to configure, run and watch simulations.

A small standard-library HTTP server bound to 127.0.0.1. It serves one page (static/app.html)
a landing page at / and the lab at /lab, plus a JSON API:

  GET  /                       landing page (static/landing.html)
  GET  /lab                    the lab: configure, run, watch (static/app.html)
  GET  /api/options              topologies, algorithms, installed Ollama models, presets
  GET  /api/runs                 past and running runs, newest first
  POST /api/runs                 {"engine": "llm"|"numeric", "config": {...}} -> {"id": ...}
  GET  /api/runs/<id>            the run's data for the viewer (works while it is running)
  GET  /api/runs/<id>/status     {"status", "day", "days", "error"}
  POST /api/runs/<id>/cancel     stop a running simulation after the current day

Each run executes in its own thread with its own asyncio loop, writing the usual run folder.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from agorasim.config import ExperimentConfig
from agorasim.fake_llm import BEHAVIOURS

OLLAMA_URL = "http://localhost:11434"
STATIC = resources.files("agorasim").joinpath("static")


@dataclass
class Job:
    id: str
    run_dir: Path
    engine: str
    days: int
    status: str = "running"  # running | done | failed | cancelled
    day: int = -1
    error: str | None = None
    started: float = field(default_factory=time.time)
    loop: asyncio.AbstractEventLoop | None = None
    task: asyncio.Task | None = None


class RunManager:
    def __init__(self, runs_root: Path):
        self.runs_root = runs_root
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ starting runs

    def start(self, engine: str, raw: dict) -> Job:
        from agorasim.runs import new_run_dir

        raw = dict(raw)
        raw["output_dir"] = str(self.runs_root)
        if engine == "numeric":
            raw.pop("models", None)
            raw.pop("scorer", None)
        cfg = ExperimentConfig.model_validate(raw)
        if engine == "llm" and not cfg.models:
            raise ValueError("pick at least one model")
        if engine == "numeric" and cfg.baseline is None:
            raise ValueError("pick a numeric algorithm")
        days = cfg.baseline.steps if engine == "numeric" else cfg.simulation.days
        run_dir = new_run_dir(cfg)
        job = Job(id=run_dir.name, run_dir=run_dir, engine=engine, days=days)
        with self.lock:
            self.jobs[job.id] = job
        target = self._run_numeric if engine == "numeric" else self._run_llm
        threading.Thread(target=target, args=(job, cfg), daemon=True).start()
        return job

    def _progress(self, job: Job):
        def cb(day: int, _metrics: dict) -> None:
            job.day = day

        return cb

    def _run_numeric(self, job: Job, cfg: ExperimentConfig) -> None:
        from agorasim.numeric import run_numeric

        try:
            run_numeric(cfg, run_dir=job.run_dir, progress=self._progress(job))
            job.status = "done"
        except Exception as e:
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"

    def _run_llm(self, job: Job, cfg: ExperimentConfig) -> None:
        from agorasim.engine import Simulation
        from agorasim.runs import RunWriter, update_meta

        loop = asyncio.new_event_loop()
        job.loop = loop
        writer = RunWriter(job.run_dir)
        t0 = time.time()
        try:
            sim = Simulation(cfg)
            job.task = loop.create_task(sim.run(writer, self._progress(job)))
            loop.run_until_complete(job.task)
            job.status = "done"
            update_meta(
                job.run_dir,
                finished=True,
                seconds=round(time.time() - t0, 1),
                llm_calls=sim.client.calls,
                cache_hits=sim.client.cache_hits,
            )
        except asyncio.CancelledError:
            job.status = "cancelled"
            update_meta(job.run_dir, cancelled=True)
        except Exception as e:
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            writer.close()
            loop.close()

    def cancel(self, run_id: str) -> bool:
        job = self.jobs.get(run_id)
        if not job or job.status != "running" or not job.loop or not job.task:
            return False
        job.loop.call_soon_threadsafe(job.task.cancel)
        return True

    # ------------------------------------------------------------------ reading runs

    def run_dir(self, run_id: str) -> Path | None:
        if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
            return None
        d = self.runs_root / run_id
        return d if (d / "meta.json").exists() else None

    def status(self, run_id: str) -> dict:
        job = self.jobs.get(run_id)
        if job:
            return {"status": job.status, "day": job.day, "days": job.days, "error": job.error}
        d = self.run_dir(run_id)
        meta = json.loads((d / "meta.json").read_text()) if d else {}
        status = (
            "cancelled"
            if meta.get("cancelled")
            else ("done" if meta.get("finished") else "incomplete")
        )
        return {"status": status, "day": None, "days": None, "error": None}

    def list(self) -> list[dict]:
        if not self.runs_root.exists():
            return []
        out = []
        for d in sorted(self.runs_root.iterdir(), reverse=True):
            meta_path = d / "meta.json"
            if not meta_path.exists() or not (d / "agents.parquet").exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            cfg = meta.get("config", {})
            engine = meta.get("engine") or (
                "numeric" if cfg.get("baseline") and not cfg.get("models") else "llm"
            )
            out.append(
                {
                    "id": d.name,
                    "name": cfg.get("name", d.name),
                    "started": meta.get("started"),
                    "engine": engine,
                    "topology": cfg.get("network", {}).get("topology"),
                    "n": cfg.get("network", {}).get("n"),
                    "models": [m["name"] for m in cfg.get("models", [])]
                    or ([f"numeric/{cfg['baseline']['model']}"] if cfg.get("baseline") else []),
                    **self.status(d.name),
                }
            )
        return out[:100]

    def payload(self, run_id: str) -> dict | None:
        from agorasim.analysis import load_run
        from agorasim.export import build_payload

        d = self.run_dir(run_id)
        if d is None:
            return None
        try:
            return build_payload(load_run(d))
        except (FileNotFoundError, ValueError, KeyError, IndexError, StopIteration):
            return None  # nothing written yet


def ollama_models() -> tuple[list[str], str | None]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1.5) as r:
            tags = json.loads(r.read())
        return sorted(f"ollama_chat/{m['name']}" for m in tags.get("models", [])), None
    except Exception:
        return [], "Ollama is not running (start it with `ollama serve`)."


def presets(config_dir: Path) -> list[dict]:
    out = []
    if not config_dir.exists():
        return out
    for p in sorted(config_dir.rglob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(raw, dict) or "network" not in raw:
            continue
        engine = "numeric" if raw.get("baseline") and not raw.get("models") else "llm"
        out.append({"file": str(p.relative_to(config_dir)), "engine": engine, "config": raw})
    return out


def make_handler(manager: RunManager, config_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "agorasim"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json")

        def _error(self, code: int, msg: str) -> None:
            self._json({"error": msg}, code)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            pages = {"/": "landing.html", "/index.html": "landing.html", "/lab": "app.html"}
            if path in pages:
                html = STATIC.joinpath(pages[path]).read_text(encoding="utf-8")
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if path == "/d3.min.js":
                return self._send(
                    200, STATIC.joinpath("d3.min.js").read_bytes(), "text/javascript; charset=utf-8"
                )
            if path == "/api/options":
                models, err = ollama_models()
                return self._json(
                    {
                        "ollama_models": models,
                        "ollama_error": err,
                        "fake_models": ["fake/bounded", *(f"fake/{b}" for b in BEHAVIOURS)],
                        "presets": presets(config_dir),
                    }
                )
            if path == "/api/runs":
                return self._json(manager.list())
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                run_id = parts[2]
                if manager.run_dir(run_id) is None:
                    return self._error(404, "run not found")
                if len(parts) == 4 and parts[3] == "status":
                    return self._json(manager.status(run_id))
                if len(parts) == 3:
                    return self._json(
                        {"status": manager.status(run_id), "data": manager.payload(run_id)}
                    )
            self._error(404, "not found")

        def do_POST(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._error(400, "invalid JSON")
            if path == "/api/runs":
                try:
                    job = manager.start(body.get("engine", "llm"), body.get("config", {}))
                except ValidationError as e:
                    msg = "; ".join(
                        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                        for err in e.errors()
                    )
                    return self._error(400, msg)
                except (ValueError, TypeError) as e:
                    return self._error(400, str(e))
                return self._json({"id": job.id}, HTTPStatus.CREATED)
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel":
                return self._json({"cancelled": manager.cancel(parts[2])})
            self._error(404, "not found")

    return Handler


def serve(
    port: int = 8765,
    runs_root: Path = Path("runs"),
    config_dir: Path = Path("configs"),
    open_browser: bool = True,
) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    manager = RunManager(runs_root)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(manager, config_dir))
    url = f"http://127.0.0.1:{port}/"
    print(f"agorasim app running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
