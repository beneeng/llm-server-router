from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

from .config import ModelConfig, ServerConfig

if TYPE_CHECKING:
    from .backends import Backend


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
VENV_DIR = STATE_DIR / "venvs"
APC_DIR = STATE_DIR / "apc"


def _backend_python(env: str) -> str:
    """Return the Python executable for an isolated backend env.

    Non-`main` backends live in isolated venvs under ~/.local/share/mlx/venvs/<env>/
    so they can install packages that conflict with the main tool venv (e.g. mflux pins
    an older mlx than mlx-vlm requires). These venvs are auto-provisioned on first model
    start, or created explicitly with: mlx install <env>
    """
    venv_python = VENV_DIR / env / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    raise FileNotFoundError(
        f"No isolated venv found for backend env '{env}' at {venv_python.parent}. "
        f"Run: mlx install {env}"
    )


def provision_env(env: str, packages: Sequence[str], log_fh: IO[str] | None = None) -> None:
    """Create the isolated venv `env` and install `packages` into it.

    Output goes to `log_fh` if given (auto-provision during a model start), else
    to the console (explicit `mlx install`). Idempotent enough: re-running just
    recreates/upgrades.
    """
    venv_dir = VENV_DIR / env
    redirect = {"stdout": log_fh, "stderr": subprocess.STDOUT} if log_fh else {}

    def _say(msg: str) -> None:
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()
        else:
            print(msg, flush=True)

    _say(f"[mlx] creating venv at {venv_dir}")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, **redirect)
    pip = venv_dir / "bin" / "pip"
    _say(f"[mlx] installing into '{env}': {' '.join(packages)}")
    from .backends import BACKENDS

    build_env = {k: v for b in BACKENDS.values() if b.env == env for k, v in b.build_env.items()}
    subprocess.run(
        [str(pip), "install", "--upgrade", *packages],
        check=True, env={**os.environ, **build_env}, **redirect,
    )
    _say(f"[mlx] backend env '{env}' ready")


def ensure_backend_env(backend: "Backend", log_fh: IO[str] | None = None) -> None:
    """Auto-provision an isolated backend venv on first use (no-op for env=='main')."""
    if backend.env == "main":
        return
    if (VENV_DIR / backend.env / "bin" / "python3").exists():
        return
    if log_fh is not None:
        log_fh.write(f"[mlx] backend env '{backend.env}' missing — provisioning...\n")
        log_fh.flush()
    provision_env(backend.env, backend.packages, log_fh=log_fh)


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
    APC_DIR.mkdir(parents=True, exist_ok=True)


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
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, we just can't signal it


def _port_bound(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


def start_model(model: ModelConfig) -> int:
    """Start an mlx_lm.server process for the model. Returns the PID."""
    _ensure_dirs()
    pids = _load_pids()

    if model.name in pids and _is_alive(pids[model.name]):
        return pids[model.name]

    if _port_bound(model.port):
        raise RuntimeError(
            f"Port {model.port} is already in use. "
            f"Stop whatever is using it before starting {model.name}."
        )

    log = _log_path(model.name)
    with log.open("a") as fh:
        # Auto-provision the isolated venv on first use (slow; logged to the model's
        # log file so `mlx logs NAME` shows progress). Blocks before launch.
        ensure_backend_env(model.resolve_backend(), log_fh=fh)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.update(model.resolve_backend().env_vars(model))
        proc = subprocess.Popen(
            model.server_args(),
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            # Never inherit the router's cwd: if that directory is deleted or
            # replaced while the router runs (re-cloned repo, removed worktree),
            # the child crashes on import (transformers calls Path("src").resolve()).
            cwd=STATE_DIR,
        )

    # Brief pause to catch immediate failures (e.g. bad args)
    time.sleep(0.3)
    if not _is_alive(proc.pid):
        raise RuntimeError(
            f"Process for {model.name} exited immediately. Check: mlx logs {model.name}"
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

    if _port_bound(cfg.port):
        raise RuntimeError(
            f"Port {cfg.port} is already in use. "
            "Run 'mlx stop --router' first, or kill whatever is on that port."
        )

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
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    time.sleep(0.3)
    if not _is_alive(proc.pid):
        raise RuntimeError(
            f"Router exited immediately. Check: mlx logs"
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
