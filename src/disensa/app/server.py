"""`disensa app`: a small local web server for the three pages and the lab's JSON API.

Standard library only (http.server), bound to 127.0.0.1.

  GET  /                        landing page       web/landing/index.html
  GET  /lab                     the lab            web/lab/index.html
  GET  /docs                    documentation      web/docs/index.html
  GET  /<folder>/<file>         any other file under web/ (css, js, vendor/d3.min.js)

  GET  /api/options             installed Ollama models, offline fake models, presets
  GET  /api/runs                past and running runs, newest first
  POST /api/runs                {"config": {...}} -> {"id": ...}   starts a run
  GET  /api/runs/<id>           the run's data for the viewer (works while it is running)
  GET  /api/runs/<id>/status    {"status", "step", "steps", "error"}
  POST /api/runs/<id>/cancel    stop a running simulation after the current step

Each run executes in its own thread with its own asyncio loop, writing the usual run folder.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import threading
import traceback
import urllib.request
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from disensa.config import ExperimentConfig
from disensa.llm.fake import BEHAVIOURS

OLLAMA_URL = "http://localhost:11434"
WEB = resources.files("disensa").joinpath("app", "web")
PAGES = {"/": "landing", "/lab": "lab", "/docs": "docs"}
TYPES = {
    ".js": "text/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".svg": "image/svg+xml",
}


@dataclass
class Job:
    id: str
    run_dir: Path
    steps: int
    status: str = "running"  # running | done | failed | cancelled
    step: int = -1
    error: str | None = None
    loop: asyncio.AbstractEventLoop | None = None
    task: asyncio.Task | None = None


class RunManager:
    def __init__(self, runs_root: Path):
        self.runs_root = runs_root
        self.jobs: dict[str, Job] = {}

    def start(self, raw: dict) -> Job:
        from disensa.runs import new_run_dir

        cfg = ExperimentConfig.model_validate({**raw, "output_dir": str(self.runs_root)})
        run_dir = new_run_dir(cfg)
        job = Job(id=run_dir.name, run_dir=run_dir, steps=cfg.steps)
        self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, cfg), daemon=True).start()
        return job

    def _run(self, job: Job, cfg: ExperimentConfig) -> None:
        from disensa.engine import run
        from disensa.runs import update_meta

        def progress(t: int, _m: dict) -> None:
            job.step = t

        loop = asyncio.new_event_loop()
        job.loop = loop
        try:
            job.task = loop.create_task(run(cfg, job.run_dir, progress))
            loop.run_until_complete(job.task)
            job.status = "done"
        except asyncio.CancelledError:
            job.status = "cancelled"
            update_meta(job.run_dir, cancelled=True)
        except Exception as e:
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            loop.close()

    def cancel(self, run_id: str) -> bool:
        job = self.jobs.get(run_id)
        if not job or job.status != "running" or not job.loop or not job.task:
            return False
        job.loop.call_soon_threadsafe(job.task.cancel)
        return True

    # ---------------------------------------------------------------- reading runs

    def run_dir(self, run_id: str) -> Path | None:
        if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
            return None
        d = self.runs_root / run_id
        return d if (d / "meta.json").exists() else None

    def status(self, run_id: str) -> dict:
        job = self.jobs.get(run_id)
        if job:
            return {"status": job.status, "step": job.step, "steps": job.steps, "error": job.error}
        d = self.run_dir(run_id)
        meta = json.loads((d / "meta.json").read_text()) if d else {}
        status = (
            "cancelled"
            if meta.get("cancelled")
            else "done"
            if meta.get("finished")
            else "incomplete"
        )
        return {"status": status, "step": None, "steps": None, "error": None}

    def list(self) -> list[dict]:
        if not self.runs_root.exists():
            return []
        out = []
        for d in sorted(self.runs_root.iterdir(), reverse=True):
            try:
                cfg = json.loads((d / "meta.json").read_text())["config"]
                dyn = cfg["dynamics"]
            except (OSError, json.JSONDecodeError, KeyError):
                continue  # not a run folder, or one from an older version
            models = (
                [f"numeric/{dyn['rule']}"]
                if dyn["type"] == "numeric"
                else [m["name"] for m in dyn["models"]]
            )
            out.append(
                {
                    "id": d.name,
                    "name": cfg["name"],
                    "started": d.name.split("_", 1)[0],
                    "engine": dyn["type"],
                    "topology": cfg["network"]["topology"],
                    "n": cfg["network"]["n"],
                    "models": models,
                    **self.status(d.name),
                }
            )
        return out[:100]

    def payload(self, run_id: str) -> dict | None:
        from disensa.app.payload import build_payload
        from disensa.runs import load_run

        d = self.run_dir(run_id)
        if d is None:
            return None
        try:
            return build_payload(load_run(d))
        except (FileNotFoundError, ValueError, KeyError, IndexError):
            return None  # nothing written yet


def ollama_models() -> tuple[list[str], str | None]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1.5) as r:
            tags = json.loads(r.read())
        return sorted(f"ollama_chat/{m['name']}" for m in tags.get("models", [])), None
    except Exception:
        return [], "Ollama is not running (start it with `ollama serve`)."


def presets(config_dir: Path) -> list[dict]:
    """Every valid YAML config under configs/, offered as a starting point in the lab."""
    out = []
    for p in sorted(config_dir.rglob("*.yaml")) if config_dir.exists() else []:
        try:
            raw = yaml.safe_load(p.read_text())
            ExperimentConfig.model_validate(raw)
        except (yaml.YAMLError, ValidationError):
            continue
        out.append({"file": str(p.relative_to(config_dir)), "config": raw})
    return out


def web_file(path: str):
    """Resolve a URL path to a file under web/, or None. Never leaves the web folder."""
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts or any(p.startswith(".") for p in parts):
        return None
    f = WEB.joinpath(*parts)
    return f if f.is_file() else None


def make_handler(manager: RunManager, config_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "disensa"

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
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in PAGES:
                html = WEB.joinpath(PAGES[path], "index.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if path == "/api/options":
                models, err = ollama_models()
                fakes = ["fake/bounded", *(f"fake/{b}" for b in BEHAVIOURS)]
                return self._json(
                    {
                        "ollama_models": models,
                        "ollama_error": err,
                        "fake_models": fakes,
                        "presets": presets(config_dir),
                    }
                )
            if path == "/api/runs":
                return self._json(manager.list())
            parts = path.strip("/").split("/")
            if len(parts) in (3, 4) and parts[:2] == ["api", "runs"]:
                run_id = parts[2]
                if manager.run_dir(run_id) is None:
                    return self._error(404, "run not found")
                if len(parts) == 4 and parts[3] == "status":
                    return self._json(manager.status(run_id))
                if len(parts) == 3:
                    return self._json(
                        {"status": manager.status(run_id), "data": manager.payload(run_id)}
                    )
            f = web_file(path)
            if f is not None:
                suffix = Path(f.name).suffix
                ctype = (
                    TYPES.get(suffix)
                    or mimetypes.guess_type(f.name)[0]
                    or "application/octet-stream"
                )
                return self._send(200, f.read_bytes(), f"{ctype}; charset=utf-8")
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
                    job = manager.start(body.get("config", {}))
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
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(RunManager(runs_root), config_dir)
    )
    url = f"http://127.0.0.1:{port}/"
    print(f"disensa app running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
