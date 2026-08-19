"""Tests for the APC disk-cache TTL sweep in mlx_server/apc_gc.py.

sweep_once() reads mlx_server.manager.APC_DIR at call time (local import), so
tests point it at a tmp_path via monkeypatch instead of touching the real
~/.local/share/mlx/apc/.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_apc_gc.py -v
"""
import os
import time

import pytest

import mlx_server.manager as manager
from mlx_server.apc_gc import sweep_once


@pytest.fixture
def apc_dir(tmp_path, monkeypatch):
    d = tmp_path / "apc"
    d.mkdir()
    monkeypatch.setattr(manager, "APC_DIR", d)
    return d


def _make_shard(path, age_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 128)
    mtime = time.time() - age_seconds
    os.utime(path, (mtime, mtime))


def test_sweep_removes_only_stale_shards(apc_dir):
    stale = apc_dir / "model-a" / "ns" / "shard_old.safetensors"
    fresh = apc_dir / "model-a" / "ns" / "shard_new.safetensors"
    _make_shard(stale, age_seconds=49 * 3600)
    _make_shard(fresh, age_seconds=1 * 3600)

    removed, freed = sweep_once(ttl_seconds=48 * 3600)

    assert removed == 1
    assert freed == 128
    assert not stale.exists()
    assert fresh.exists()


def test_dry_run_deletes_nothing(apc_dir):
    stale = apc_dir / "model-a" / "ns" / "shard_old.safetensors"
    _make_shard(stale, age_seconds=49 * 3600)

    removed, freed = sweep_once(ttl_seconds=48 * 3600, dry_run=True)

    assert removed == 1
    assert freed == 128
    assert stale.exists()


def test_sweep_on_missing_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "APC_DIR", tmp_path / "does-not-exist")

    removed, freed = sweep_once()

    assert (removed, freed) == (0, 0)


def test_sweep_ignores_non_shard_files(apc_dir):
    other = apc_dir / "model-a" / "ns" / "notes.txt"
    _make_shard(other, age_seconds=49 * 3600)

    removed, freed = sweep_once(ttl_seconds=48 * 3600)

    assert removed == 0
    assert other.exists()
