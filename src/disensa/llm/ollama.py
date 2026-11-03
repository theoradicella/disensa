"""Local models through Ollama: start the server if needed, list models, download them.

disensa only uses models that run on the same machine, through Ollama
(https://ollama.com). The app and `disensa run` start `ollama serve` themselves when Ollama
is installed but not running, and stop it again on exit if they were the ones to start it.
"""

from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"

# Offered for one-click download in the lab; any other model works with `ollama pull <name>`.
SUGGESTED = [
    {"name": "llama3.2:3b", "label": "Llama 3.2 3B", "by": "Meta", "size": "2.0 GB"},
    {"name": "qwen3:4b", "label": "Qwen3 4B", "by": "Alibaba", "size": "2.5 GB"},
]

NOT_INSTALLED = "Ollama is not installed. Install it from ollama.com, then restart disensa."
NOT_STARTED = "Ollama is installed but could not be started. Run `ollama serve` in a terminal."

_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def is_running(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def binary() -> str | None:
    """The `ollama` command, or the one inside the macOS app if it was never linked."""
    found = shutil.which("ollama")
    if found:
        return found
    mac_app = Path("/Applications/Ollama.app/Contents/Resources/ollama")
    return str(mac_app) if mac_app.exists() else None


def ensure_running(wait: float = 10.0) -> str | None:
    """Start `ollama serve` if it is not running. Returns None when ready, else a message."""
    global _proc
    if is_running():
        return None
    exe = binary()
    if exe is None:
        return NOT_INSTALLED
    with _lock:
        if _proc is None or _proc.poll() is not None:
            _proc = subprocess.Popen(
                [exe, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            atexit.register(stop)
    deadline = time.time() + wait
    while time.time() < deadline:
        if is_running(timeout=0.5):
            return None
        time.sleep(0.3)
    return NOT_STARTED


def stop() -> None:
    """Stop the Ollama server, only if disensa started it."""
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()


def installed() -> list[str]:
    """Names of the models Ollama has, e.g. ["llama3.2:3b"]."""
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1.5) as r:
        tags = json.loads(r.read())
    return sorted(m["name"] for m in tags.get("models", []))


class Downloads:
    """Model downloads (`ollama pull`) running in the background, with their progress."""

    def __init__(self) -> None:
        self._state: dict[str, dict] = {}
        self._cancel: set[str] = set()
        self._lock = threading.Lock()

    def status(self) -> dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._state.items()}

    def start(self, name: str) -> None:
        with self._lock:
            if self._state.get(name, {}).get("status") == "downloading":
                return
            self._state[name] = {"status": "downloading", "progress": 0.0, "error": None}
            self._cancel.discard(name)
        threading.Thread(target=self._pull, args=(name,), daemon=True).start()

    def cancel(self, name: str) -> None:
        """Stop a download; closing the connection makes Ollama stop pulling."""
        with self._lock:
            if self._state.get(name, {}).get("status") == "downloading":
                self._cancel.add(name)

    def _set(self, name: str, **kw) -> None:
        with self._lock:
            self._state[name].update(kw)

    def _pull(self, name: str) -> None:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/pull",
            data=json.dumps({"model": name, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        layers: dict[str, tuple[int, int]] = {}  # digest -> (completed, total)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                for line in r:
                    if name in self._cancel:
                        self._set(name, status="cancelled", progress=0.0)
                        return
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    if ev.get("error"):
                        raise RuntimeError(ev["error"])
                    if ev.get("digest") and ev.get("total"):
                        layers[ev["digest"]] = (ev.get("completed", 0), ev["total"])
                        done = sum(c for c, _ in layers.values())
                        total = sum(t for _, t in layers.values())
                        self._set(name, progress=done / total)
                    if ev.get("status") == "success":
                        self._set(name, status="done", progress=1.0)
                        return
            raise RuntimeError("the download stopped before finishing")
        except Exception as e:
            self._set(name, status="failed", error=str(e))
