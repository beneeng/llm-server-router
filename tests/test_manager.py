"""Tests for subprocess launching in mlx_server/manager.py.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_manager.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mlx_server import manager
from mlx_server.config import ModelConfig


def _fake_start(monkeypatch, tmp_path: Path) -> dict:
    """Run start_model() with every side effect faked; return Popen's kwargs."""
    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(argv, **kwargs):
        captured.update(kwargs)
        return FakeProc()

    log = tmp_path / "model.log"
    monkeypatch.setattr(manager, "_ensure_dirs", lambda: None)
    monkeypatch.setattr(manager, "_load_pids", lambda: {})
    monkeypatch.setattr(manager, "_save_pids", lambda pids: None)
    monkeypatch.setattr(manager, "_port_bound", lambda port: False)
    monkeypatch.setattr(manager, "_is_alive", lambda pid: True)
    monkeypatch.setattr(manager, "_log_path", lambda name: log)
    monkeypatch.setattr(manager, "ensure_backend_env", lambda backend, log_fh=None: None)
    monkeypatch.setattr(manager.time, "sleep", lambda s: None)
    monkeypatch.setattr(manager.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ModelConfig, "server_args", lambda self: ["python", "wrapper.py"])
    return captured


def test_backend_gets_a_stable_working_directory(monkeypatch, tmp_path):
    """The backend must not inherit the router's cwd. If that directory is later
    deleted or replaced (re-cloned repo), the child dies at import time:
    transformers calls Path("src").resolve() -> os.getcwd() -> FileNotFoundError."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(manager, "STATE_DIR", state_dir)
    captured = _fake_start(monkeypatch, tmp_path)

    manager.start_model(ModelConfig(name="m", source="s", backend="mlx-lm"))

    assert captured["cwd"] == state_dir


def test_child_survives_when_parent_cwd_was_deleted(tmp_path):
    """End to end: with cwd=<existing dir> the child can resolve relative paths
    even though the parent's own cwd no longer exists."""
    gone = tmp_path / "gone"
    gone.mkdir()
    stable = tmp_path / "stable"
    stable.mkdir()
    code = "from pathlib import Path; print(Path('src').resolve())"
    saved = os.getcwd()
    os.chdir(gone)
    gone.rmdir()
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", code], cwd=stable, capture_output=True, text=True
        )
    finally:
        os.chdir(saved)
    assert result.returncode == 0, result.stderr
