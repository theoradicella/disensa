"""`disensa app`: a small local web server for the three pages and the lab's JSON API.

Standard library only (http.server), bound to 127.0.0.1.

  GET  /                        landing page       web/landing/index.html
  GET  /lab                     the lab            web/lab/index.html
  GET  /docs                    documentation      web/docs/index.html
  GET  /<folder>/<file>         any other file under web/ (css, js, vendor/d3.min.js)

  GET  /api/options             installed Ollama models, suggested downloads, download progress
  GET  /api/models/pull         {"<model>": {"status", "progress", "error"}}  download progress
  POST /api/models/pull         {"name": "<model>"}  start downloading a suggested model
  POST /api/models/cancel       {"name": "<model>"}  stop that download
  GET  /api/runs                past and running runs, newest first; runs of a sweep
                                (runs/<sweep>/<run>/) have the id "<sweep>/<run>"
                                (URL-encoded in the paths below) and a "sweep" field
  POST /api/runs                {"config": {...}} -> {"id": ...}   starts a run
  POST /api/check               {"config": {...}} -> {"checks": [...]}  warnings (checks.py)
  GET  /api/runs/<id>           the run's data for the viewer (works while it is running)
  GET  /api/runs/<id>/status    {"status", "step", "steps", "error", "agents", "replies",
                                 "failed", "elapsed"}  (live counters while running)
  POST /api/runs/<id>/cancel    stop a running simulation now (finished steps are kept)
  DELETE /api/runs/<id>         delete a run folder (not while it is running)

Each run executes in its own thread with its own asyncio loop, writing the usual run folder.
Ollama is started on launch if it is installed but not running (llm/ollama.py).
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import shutil
import threading
import time
import traceback
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from pydantic import ValidationError

from disensa.config import ExperimentConfig
from disensa.llm import ollama

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
    step: int = -1  # last finished step; -1 until the first messages (step 0) are in
    agents: int = 0
    replies: int = 0  # replies received for the step in progress
    failed: int = 0  # of those, how many were malformed / errored
    started: float = 0.0
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
        job = Job(
            id=run_dir.name,
            run_dir=run_dir,
            steps=cfg.steps,
            agents=cfg.network.n,
            started=time.time(),
        )
        self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, cfg), daemon=True).start()
        return job

    def _run(self, job: Job, cfg: ExperimentConfig) -> None:
        from disensa.engine import RunFailed, run

        llm = cfg.dynamics.type == "llm"

        def log(msg: str) -> None:
            print(f"[{job.id}] {msg}  [{time.time() - job.started:.0f}s]", flush=True)

        def progress(t: int, m: dict) -> None:
            job.step, job.replies, job.failed = t, 0, 0
            what = "first messages written" if t == 0 else f"step {t}/{job.steps} done"
            log(
                f"{what}  pol {m['polarization']:.3f}  dis {m['global_disagreement']:.3f}"
                + (f"  errors {m['errors']}" if llm else "")
            )

        def on_reply(t: int, _agent: int, err: str | None) -> None:
            job.replies += 1
            job.failed += err is not None
            if err:
                log(f"step {t}: agent {_agent} reply failed: {err[:160]}")

        what = (
            f"numeric/{cfg.dynamics.rule}"
            if not llm
            else ", ".join(m.name for m in cfg.dynamics.models)
        )
        log(f"started: {cfg.network.n} agents, {cfg.steps} steps, {what}")
        loop = asyncio.new_event_loop()
        job.loop = loop
        try:
            job.task = loop.create_task(run(cfg, job.run_dir, progress, on_reply))
            loop.run_until_complete(job.task)
            job.status = "done"
            log("finished")
        except asyncio.CancelledError:
            job.status = "cancelled"  # the engine has marked meta.json already
            log("cancelled")
        except RunFailed as e:
            job.status, job.error = "failed", e.reason
            log(f"stopped: {e.reason}")
        except Exception as e:
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            log(f"failed: {job.error}")
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
        """The folder of a run id: "<run>" or, inside a sweep folder, "<sweep>/<run>"."""
        parts = (run_id or "").split("/")
        if not 1 <= len(parts) <= 2 or "\\" in run_id:
            return None
        if any(not p or p.startswith(".") for p in parts):
            return None
        d = self.runs_root.joinpath(*parts)
        return d if (d / "meta.json").exists() else None

    def status(self, run_id: str) -> dict:
        job = self.jobs.get(run_id)
        if job:
            return {
                "status": job.status,
                "step": job.step,
                "steps": job.steps,
                "error": job.error,
                "agents": job.agents,
                "replies": job.replies,
                "failed": job.failed,
                "elapsed": round(time.time() - job.started, 1),
            }
        d = self.run_dir(run_id)
        meta = json.loads((d / "meta.json").read_text()) if d else {}
        return {
            "status": meta_status(meta),
            "step": None,
            "steps": None,
            "error": meta.get("reason"),
        }

    def delete(self, run_id: str) -> bool:
        """Remove a run folder from disk. Refused while the run is going."""
        job = self.jobs.get(run_id)
        if job and job.status == "running":
            return False
        d = self.run_dir(run_id)
        if d is None:
            return False
        shutil.rmtree(d)
        self.jobs.pop(run_id, None)
        return True

    def list(self) -> list[dict]:
        """Runs newest first: the last 100 runs at the top level, and every run of a sweep."""
        if not self.runs_root.exists():
            return []
        top, swept = [], []
        for d in sorted(self.runs_root.iterdir(), reverse=True):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if (d / "meta.json").exists():
                if len(top) < 100 and (e := self._entry(d, d.name)):
                    top.append(e)
            elif (d / "sweep.json").exists() or any(d.glob("*/meta.json")):
                for r in sorted(d.iterdir(), reverse=True):
                    if (r / "meta.json").exists() and (e := self._entry(r, f"{d.name}/{r.name}")):
                        swept.append({**e, "sweep": d.name})
        return top + swept

    def _entry(self, d: Path, run_id: str) -> dict | None:
        """The list entry of one run folder, or None if it is not one."""
        try:
            meta = json.loads((d / "meta.json").read_text())
            cfg = meta["config"]
            dyn = cfg["dynamics"]
        except (OSError, json.JSONDecodeError, KeyError):
            return None  # not a run folder, or one from an older version
        models = (
            [f"numeric/{dyn['rule']}"]
            if dyn["type"] == "numeric"
            else [m["name"] for m in dyn["models"]]
        )
        return {
            **self.status(run_id),  # first, so its "steps": None does not hide the config's
            "id": run_id,
            "name": cfg["name"],
            "started": d.name.split("_", 1)[0],
            "engine": dyn["type"],
            "topology": cfg["network"]["topology"],
            "n": cfg["network"]["n"],
            "steps": cfg.get("steps"),
            "seed": cfg.get("seed"),
            "replicate": cfg.get("replicate", 0),
            "seconds": meta.get("seconds"),  # how long it ran; not saved by older versions
            "failed_pct": (meta.get("report") or {}).get("failed_pct"),  # None: not recorded
            "models": models,
            "config": cfg,  # for "Duplicate" in the lab
        }

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


def meta_status(meta: dict) -> str:
    """The lab's status of a run that is not going in this server: done | cancelled | failed |
    incomplete. A run still marked "running" was stopped by a crash or a closed terminal."""
    status = meta.get("status")
    if status == "finished" or (status is None and meta.get("finished")):
        return "done"
    if status in ("cancelled", "failed"):
        return status
    if status is None and meta.get("cancelled"):  # runs from before 0.3
        return "cancelled"
    return "incomplete"


DOWNLOADS = ollama.Downloads()


def ollama_models() -> tuple[list[str], str | None]:
    """Installed models as LiteLLM names ("ollama_chat/<name>"), or an error message."""
    err = ollama.ensure_running(wait=3.0)
    if err:
        return [], err
    try:
        return [f"ollama_chat/{m}" for m in ollama.installed()], None
    except Exception:
        return [], ollama.NOT_STARTED


def web_file(path: str):
    """Resolve a URL path to a file under web/, or None. Never leaves the web folder."""
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts or any(p.startswith(".") for p in parts):
        return None
    f = WEB.joinpath(*parts)
    return f if f.is_file() else None


def make_handler(manager: RunManager):
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
                return self._json(
                    {
                        "ollama_models": models,
                        "ollama_error": err,
                        "ollama_missing": err == ollama.NOT_INSTALLED,
                        "suggested": ollama.SUGGESTED,
                        "downloads": DOWNLOADS.status(),
                    }
                )
            if path == "/api/models/pull":
                return self._json(DOWNLOADS.status())
            if path == "/api/runs":
                return self._json(manager.list())
            parts = [urllib.parse.unquote(p) for p in path.strip("/").split("/")]
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
            if path == "/api/models/pull":
                name = body.get("name")
                if name not in {m["name"] for m in ollama.SUGGESTED}:
                    return self._error(400, "unknown model")
                if err := ollama.ensure_running():
                    return self._error(503, err)
                DOWNLOADS.start(name)
                return self._json(DOWNLOADS.status(), HTTPStatus.ACCEPTED)
            if path == "/api/models/cancel":
                DOWNLOADS.cancel(body.get("name"))
                return self._json(DOWNLOADS.status())
            if path == "/api/check":
                from disensa.checks import raw_checks

                return self._json({"checks": raw_checks(body.get("config") or {})})
            parts = [urllib.parse.unquote(p) for p in path.strip("/").split("/")]
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel":
                return self._json({"cancelled": manager.cancel(parts[2])})
            self._error(404, "not found")

        def do_DELETE(self):  # noqa: N802
            parts = [
                urllib.parse.unquote(p) for p in self.path.split("?", 1)[0].strip("/").split("/")
            ]
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                if manager.delete(parts[2]):
                    return self._json({"deleted": parts[2]})
                return self._error(409, "this run is still going or does not exist")
            self._error(404, "not found")

    return Handler


def serve(
    port: int = 8765,
    runs_root: Path = Path("runs"),
    open_browser: bool = True,
) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(RunManager(runs_root)))
    url = f"http://127.0.0.1:{port}/"
    if not ollama.is_running():
        print("Starting Ollama…")
        err = ollama.ensure_running()
        if err:
            print(f"  {err}  (numeric runs still work)")
    print(f"disensa app running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ollama.stop()
