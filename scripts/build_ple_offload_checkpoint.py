#!/usr/bin/env python3
"""
Build an mlx-vlm-native, disk-offloaded-PLE checkpoint for Qwen3.8-Flash-Next.

Background: `mlx-vlm` >=0.7.0rc0 can keep the qwen4_exp architecture's ~30GB
(at Q4) PLE n-gram embedding table on disk instead of loading it resident,
via `mlx_vlm.models.qwen4_exp.ple_storage.prepare_external_ple_model`. That
function ships with no CLI wired up — this script is the missing CLI. It:

  1. Downloads a source checkpoint (default: mlx-community/Qwen3.8-Flash-Next-4bit)
  2. Verifies its PLE shards are in a quantization layout ple_storage.py
     understands (4-bit/group-32/affine or 4-bit/group-16/nvfp4)
  3. Builds a hard-linked "external PLE" view of it (near-zero extra disk,
     near-instant — no re-quantization, no GPU compute)
  4. By default, also materializes the PLE table into one contiguous
     mmap-friendly file for faster cold/random row access (~30GB extra
     disk, a few minutes of I/O) — skip with --no-materialize

MUST be run under the tool venv Python (where mlx-vlm is installed):
  ~/.local/share/uv/tools/mlx-server/bin/python scripts/build_ple_offload_checkpoint.py

Usage:
  python scripts/build_ple_offload_checkpoint.py
  python scripts/build_ple_offload_checkpoint.py --no-materialize
  python scripts/build_ple_offload_checkpoint.py --source-repo <other-repo>

Note on the unmaterialized layout: `prepare_external_ple_model` does not
copy the PLE table's bytes — the output checkpoint's manifest points back
at the source checkpoint's safetensors files by relative path. Do not move
or delete --source-dir after conversion unless you ran the materialize step
(which does copy the row data, making --dest-dir self-contained).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

CACHE_ROOT = Path.home() / ".cache" / "mlx-vlm"
DEFAULT_SOURCE_REPO = "mlx-community/Qwen3.8-Flash-Next-4bit"

# Rough preflight thresholds. The source checkpoint is ~112GB; materializing
# copies the ~30GB PLE table into the target dir on top of that.
MIN_FREE_GB_DOWNLOAD = 120
MIN_FREE_GB_MATERIALIZE = 35


def step(title: str) -> None:
    print(f"\n{'─' * 60}\n{title}\n{'─' * 60}")


def _free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024**3)


def _dir_size_gb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024**3)


def download_source(source_repo: str, source_dir: Path) -> None:
    step(f"Step 1: Download source checkpoint ({source_repo})")

    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        print(f"✓ Source checkpoint already present at {source_dir}")
        return

    free_gb = _free_gb(source_dir.parent)
    if free_gb < MIN_FREE_GB_DOWNLOAD:
        print(
            f"✗ Only {free_gb:.1f}GB free at {source_dir.parent}; "
            f"need ~{MIN_FREE_GB_DOWNLOAD}GB for the source checkpoint."
        )
        sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub not installed in current env.")
        print("  Install with: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {source_repo} → {source_dir} (this is ~112GB, will take a while)")
    snapshot_download(repo_id=source_repo, local_dir=str(source_dir))
    print(f"✓ Source checkpoint downloaded to {source_dir}")


def verify_ple_layout(source_dir: Path) -> None:
    step("Step 2: Verify PLE quantization layout")

    from mlx_vlm.models.qwen4_exp import ple_storage

    with tempfile.TemporaryDirectory() as tmp:
        scratch_manifest = Path(tmp) / "scratch-manifest.json"
        try:
            manifest = ple_storage.build_quantized_ple_manifest(
                source_dir, scratch_manifest
            )
        except ValueError as e:
            print(f"✗ Source checkpoint is not compatible with external PLE storage:\n  {e}")
            print(
                "  This means its PLE shards are not quantized as 4-bit/group-32/affine "
                "or 4-bit/group-16/nvfp4 — the two layouts ple_storage.py understands. "
                "Re-quantizing with a current mlx_vlm.convert run should fix this "
                "(qwen4_exp's own quant_predicate forces the compatible layout), "
                "but that's real GPU work, not a data-layout job."
            )
            sys.exit(1)

    quant = manifest["quantization"]
    print(
        f"✓ PLE shards are {quant['bits']}-bit / group-{quant['group_size']} / "
        f"{quant['mode']} — compatible. {manifest['row_count']:,} rows, "
        f"row_width={manifest['row_width']}."
    )


def convert(source_dir: Path, target_dir: Path, materialize: bool) -> None:
    step("Step 3: Build external-PLE checkpoint view")

    from mlx_vlm.models.qwen4_exp import ple_storage

    manifest_path = target_dir / "ple-store.json"
    if manifest_path.exists():
        print(f"✓ External-PLE view already built at {target_dir}")
    else:
        provenance = ple_storage.prepare_external_ple_model(source_dir, target_dir)
        print(f"✓ Hard-linked {len(provenance['linked_weight_files'])} weight file(s) into {target_dir}")
        print(f"✓ Wrote manifest → {manifest_path}")

    if not materialize:
        print(
            "  Skipping materialize (--no-materialize) — target still points back "
            f"at {source_dir} for PLE row data. Do not move/delete the source dir."
        )
        return

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("layout") == "interleaved_q4":
        print("✓ PLE table already materialized (interleaved_q4 layout).")
        return

    free_gb = _free_gb(target_dir)
    if free_gb < MIN_FREE_GB_MATERIALIZE:
        print(
            f"✗ Only {free_gb:.1f}GB free at {target_dir}; "
            f"need ~{MIN_FREE_GB_MATERIALIZE}GB to materialize the PLE table."
        )
        sys.exit(1)

    print("Materializing PLE table into one contiguous file (this copies ~30GB, a few minutes)…")
    # NOTE: model_path here is the *source* checkpoint, not target_dir — the
    # range manifest built so far only has byte offsets into source_dir's
    # safetensors files, and materialize needs to actually read those bytes.
    ple_storage.materialize_interleaved_ple_store(source_dir, manifest_path)
    print(f"✓ Materialized PLE row store → {target_dir / 'ple-q4.rows'}")
    print(
        f"  {target_dir} is now self-contained — {source_dir} can be deleted "
        "to reclaim disk space if you don't need it for anything else "
        "(not done automatically by this script)."
    )


def verify_output(source_dir: Path, target_dir: Path) -> None:
    step("Step 4: Verify output + next steps")

    config = json.loads((target_dir / "config.json").read_text())
    ple_cfg = config.get("text_config", {}).get("ple_storage")
    if not ple_cfg or not ple_cfg.get("manifest"):
        print("✗ target config.json is missing text_config.ple_storage.manifest — something went wrong.")
        sys.exit(1)
    if not (target_dir / "EXTERNAL_PLE.json").exists():
        print("✗ EXTERNAL_PLE.json provenance file missing — something went wrong.")
        sys.exit(1)
    print("✓ config.json + provenance file both present and well-formed.")

    source_gb = _dir_size_gb(source_dir)
    target_gb = _dir_size_gb(target_dir)
    print(f"  Source checkpoint (apparent size): {source_gb:.1f}GB")
    print(f"  Target checkpoint (apparent size): {target_gb:.1f}GB")
    print(
        "  (apparent sizes double-count hard-linked bytes shared between the two "
        "dirs — actual incremental disk usage for the target is much smaller "
        "than the number above unless materialized)"
    )

    print("\nNext steps:")
    print("  1. Smoke test (run under the tool venv python):")
    print(
        f'     mlx_vlm.generate --model "{target_dir}" '
        '--prompt "What is 2+2? Answer with just the number." --max-tokens 10'
    )
    print("     Confirm it outputs 4, not garbage.")
    print("  2. Add an entry to ~/.config/mlx/models.yaml pointing source: at:")
    print(f"       {target_dir}")
    print("     (see the qwen3.8-flash-next-ple template already discussed)")
    print("  3. cp config/models.example.yaml ~/.config/mlx/models.yaml && mlx stop && mlx start")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    p.add_argument(
        "--source-dir",
        type=Path,
        default=CACHE_ROOT / "qwen3.8-flash-next-4bit-source",
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        default=CACHE_ROOT / "qwen3.8-flash-next-ple-offload",
    )
    p.add_argument(
        "--no-materialize",
        action="store_true",
        help="Skip copying the PLE table into one contiguous file (default: materialize)",
    )
    args = p.parse_args()

    try:
        import mlx_vlm.models.qwen4_exp.ple_storage  # noqa: F401
    except ImportError:
        print(
            "✗ mlx_vlm.models.qwen4_exp.ple_storage not importable.\n"
            "  This script needs mlx-vlm>=0.7.0rc0. Run under the tool venv:\n"
            "  ~/.local/share/uv/tools/mlx-server/bin/python "
            "scripts/build_ple_offload_checkpoint.py"
        )
        sys.exit(1)

    print("=== Qwen3.8-Flash-Next PLE-offload checkpoint builder ===")

    download_source(args.source_repo, args.source_dir)
    verify_ple_layout(args.source_dir)
    convert(args.source_dir, args.dest_dir, materialize=not args.no_materialize)
    verify_output(args.source_dir, args.dest_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
