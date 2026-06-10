from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import ModelConfig, ServerConfig


def _bin(name: str) -> str:
    """Locate an executable: venv bin dir first, then PATH."""
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"'{name}' not found. Make sure it is installed in the same environment as mlx-server."
    )

STATE_DIR = Path("~/.local/share/mlx").expanduser()
PID_FILE = STATE_DIR / "pids.json"
LOG_DIR = STATE_DIR / "logs"

ROUTER_NAME = "__router__"


@dataclass
class ProcessStatus:
    name: str
    pid: int | None
    alive: bool
    port: int | None
    log: Path | None


def _ensure_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _load_pids() -> dict[str, int]:
    if PID_FILE.exists():
        return json.loads(PID_FILE.read_text())
    return {}


def _save_pids(pids: dict[str, int]) -> None:
    PID_FILE.write_text(json.dumps(pids, indent=2))


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


def start_model(model: ModelConfig) -> int:
    """Start a vllm-mlx process for the model. Returns the PID."""
    _ensure_dirs()
    pids = _load_pids()

    if model.name in pids and _is_alive(pids[model.name]):
        return pids[model.name]

    log = _log_path(model.name)
    with log.open("a") as fh:
        proc = subprocess.Popen(
            model.vllm_args(),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pids[model.name] = proc.pid
    _save_pids(pids)
    return proc.pid


def start_router(cfg: ServerConfig) -> int:
    """Start the FastAPI router process. Returns the PID."""
    _ensure_dirs()
    pids = _load_pids()

    if ROUTER_NAME in pids and _is_alive(pids[ROUTER_NAME]):
        return pids[ROUTER_NAME]

    log = _log_path(ROUTER_NAME)
    with log.open("a") as fh:
        proc = subprocess.Popen(
            [
                _bin("uvicorn"),
                "mlx_server.router:app",
                "--host", cfg.host,
                "--port", str(cfg.port),
                "--log-level", "info",
            ],
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pids[ROUTER_NAME] = proc.pid
    _save_pids(pids)
    return proc.pid


def stop_process(name: str) -> bool:
    """SIGTERM a process by name. Returns True if it was running."""
    pids = _load_pids()
    pid = pids.get(name)
    if pid is None or not _is_alive(pid):
        pids.pop(name, None)
        _save_pids(pids)
        return False

    os.kill(pid, signal.SIGTERM)
    # Give it 5 s to exit gracefully, then SIGKILL
    for _ in range(50):
        time.sleep(0.1)
        if not _is_alive(pid):
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    pids.pop(name, None)
    _save_pids(pids)
    return True


def status_all(cfg: ServerConfig) -> list[ProcessStatus]:
    pids = _load_pids()
    results: list[ProcessStatus] = []

    for model in cfg.models:
        pid = pids.get(model.name)
        results.append(ProcessStatus(
            name=model.name,
            pid=pid,
            alive=_is_alive(pid) if pid else False,
            port=model.port,
            log=_log_path(model.name),
        ))

    # Router last
    pid = pids.get(ROUTER_NAME)
    results.append(ProcessStatus(
        name=ROUTER_NAME,
        pid=pid,
        alive=_is_alive(pid) if pid else False,
        port=cfg.port,
        log=_log_path(ROUTER_NAME),
    ))
    return results
