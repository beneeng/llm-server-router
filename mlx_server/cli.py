"""
mlx — personal MLX LLM server CLI

Commands:
  mlx start  [--model NAME]   start all models (or one) + router
  mlx stop   [--model NAME]   stop all (or one); stops router first on full stop
  mlx status                  show table of all processes
  mlx logs   [MODEL]          tail the log for a model (default: router)
  mlx models                  list models from config
"""
from __future__ import annotations

import subprocess
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .manager import (
    ROUTER_NAME,
    ProcessStatus,
    start_model,
    start_router,
    status_all,
    stop_process,
)

app = typer.Typer(help="Manage local MLX LLM processes", add_completion=False)
console = Console()


# ── helpers ────────────────────────────────────────────────────────────────────

def _status_color(ps: ProcessStatus) -> str:
    return "[green]running[/green]" if ps.alive else "[red]stopped[/red]"


def _wait_for_port(port: int, timeout: float = 30.0) -> bool:
    """Poll until something binds to port, or timeout."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


# ── commands ───────────────────────────────────────────────────────────────────

@app.command()
def start(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Start only this model"),
    router: bool = typer.Option(False, "--router", help="Start only the router"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for ports to be ready"),
) -> None:
    """Start model servers and the router."""
    cfg = load_config()

    if router:
        pid = start_router(cfg)
        console.print(f"  [cyan]{ROUTER_NAME}[/cyan]  pid={pid}  port={cfg.port}")
        if wait:
            if _wait_for_port(cfg.port):
                console.print(f"  [green]✓[/green] router ready at http://{cfg.host}:{cfg.port}")
            else:
                console.print(f"  [yellow]⚠[/yellow] router port {cfg.port} not responding — check logs")
        return

    targets = [
        m for m in cfg.models
        if (model is None or m.name == model) and (model is not None or not m.lazy)
    ]
    if model and not targets:
        console.print(f"[red]Unknown model: {model}[/red]")
        raise typer.Exit(1)

    for m in targets:
        pid = start_model(m)
        console.print(f"  [cyan]{m.name}[/cyan]  pid={pid}  port={m.port}")
        if wait:
            if _wait_for_port(m.port):
                console.print(f"  [green]✓[/green] {m.name} ready")
            else:
                console.print(f"  [yellow]⚠[/yellow] {m.name} port {m.port} not responding after 30s — check logs")

    # Only start the router when starting all models (not when --model is given)
    if model is None:
        pid = start_router(cfg)
        console.print(f"  [cyan]{ROUTER_NAME}[/cyan]  pid={pid}  port={cfg.port}")
        if wait:
            if _wait_for_port(cfg.port):
                console.print(f"  [green]✓[/green] router ready at http://{cfg.host}:{cfg.port}")
            else:
                console.print(f"  [yellow]⚠[/yellow] router port {cfg.port} not responding — check logs")


@app.command()
def stop(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Stop only this model"),
    router: bool = typer.Option(False, "--router", help="Stop only the router"),
) -> None:
    """Stop model servers and the router."""
    cfg = load_config()

    if router:
        if stop_process(ROUTER_NAME):
            console.print(f"  stopped [cyan]{ROUTER_NAME}[/cyan]")
        else:
            console.print(f"  [yellow]{ROUTER_NAME} was not running[/yellow]")
        return

    if model is None:
        # Stop router first so it stops accepting traffic
        if stop_process(ROUTER_NAME):
            console.print(f"  stopped [cyan]{ROUTER_NAME}[/cyan]")
        for m in cfg.models:
            if stop_process(m.name):
                console.print(f"  stopped [cyan]{m.name}[/cyan]")
    else:
        if cfg.model_by_name(model) is None:
            console.print(f"[red]Unknown model: {model}[/red]")
            raise typer.Exit(1)
        if stop_process(model):
            console.print(f"  stopped [cyan]{model}[/cyan]")
        else:
            console.print(f"  [yellow]{model} was not running[/yellow]")


@app.command()
def status() -> None:
    """Show the state of all processes."""
    cfg = load_config()
    statuses = status_all(cfg)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("PID", justify="right")
    table.add_column("Port", justify="right")
    table.add_column("Log")

    for ps in statuses:
        table.add_row(
            ps.name,
            _status_color(ps),
            str(ps.pid) if ps.pid else "—",
            str(ps.port) if ps.port else "—",
            str(ps.log) if ps.log else "—",
        )
    console.print(table)


@app.command()
def logs(
    model: Optional[str] = typer.Argument(None, help="Model name (default: router)"),
) -> None:
    """Tail the log file for a model or the router."""
    from .manager import LOG_DIR, _log_path

    name = model or ROUTER_NAME
    log = _log_path(name)
    if not log.exists():
        console.print(f"[yellow]No log yet at {log}[/yellow]")
        raise typer.Exit(1)

    try:
        subprocess.run(["tail", "-f", str(log)])
    except KeyboardInterrupt:
        pass


@app.command(name="memory")
def memory_cmd() -> None:
    """Show per-model RSS memory usage."""
    import psutil
    from .manager import _is_alive, _load_pids

    cfg = load_config()
    pids = _load_pids()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("PID", justify="right")
    table.add_column("RSS (GB)", justify="right")
    table.add_column("Protected")

    total_gb = 0.0
    for m in cfg.models:
        pid = pids.get(m.name)
        alive = bool(pid and _is_alive(pid))
        rss_gb = 0.0
        if alive and pid:
            try:
                rss_gb = psutil.Process(pid).memory_info().rss / (1024 ** 3)
                total_gb += rss_gb
            except psutil.NoSuchProcess:
                alive = False
        table.add_row(
            m.name,
            "[green]running[/green]" if alive else "[red]stopped[/red]",
            str(pid) if pid else "—",
            f"{rss_gb:.2f}" if alive else "—",
            "✓" if m.protected else "",
        )

    free_gb = psutil.virtual_memory().available / (1024 ** 3)
    cap = cfg.max_total_memory_gb
    cap_str = f" / {cap:.1f} GB cap" if cap else ""
    floor = cfg.min_free_memory_gb
    console.print(table)
    console.print(f"\nTotal model RSS: [bold]{total_gb:.2f} GB[/bold]{cap_str}   Free RAM: [bold]{free_gb:.1f} GB[/bold] (floor {floor:.1f} GB)")
    if cap and total_gb > cap:
        console.print(f"[yellow]Over cap by {total_gb - cap:.2f} GB — eviction pending on next request[/yellow]")
    if free_gb < floor:
        console.print(f"[red]Free RAM {free_gb:.1f} GB below floor {floor:.1f} GB — new model starts will be refused[/red]")


@app.command(name="models")
def list_models() -> None:
    """List models defined in the config."""
    cfg = load_config()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Source")
    table.add_column("Port", justify="right")
    table.add_column("Lazy")
    table.add_column("Multimodal")
    table.add_column("Reasoning parser")
    table.add_column("Defaults (temp / top_p / top_k / rep_pen / max_tok)")

    for m in cfg.models:
        d = m.defaults
        table.add_row(
            m.name,
            m.source,
            str(m.port),
            "✓" if m.lazy else "",
            "✓" if m.multimodal else "",
            m.reasoning_parser or "",
            f"{d.temperature} / {d.top_p} / {d.top_k} / {d.repetition_penalty} / {d.max_tokens}",
        )
    console.print(table)
