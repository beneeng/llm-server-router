#!/usr/bin/env python3
"""
Build a disk-offloaded-PLE checkpoint for a fine-tuned/abliterated Qwen3.8-Flash-Next
variant by reusing the PLE store of an already built offload checkpoint.

Why: variants such as orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX leave the PLE
n-gram embedding table untouched (verified: dequantized base rows match the
variant's BF16 rows with ~8.5 % relative error = plain 4-bit noise), but ship it
as BF16 (~102 GB), which `ple_storage.py` rejects (it only reads 4-bit/g32/affine
or 4-bit/g16/nvfp4). The offload checkpoint never needs the PLE bytes in its
safetensors anyway - they live in `ple-store.json` + `ple-q4.rows`. So:

  1. Download only the NON-PLE tensors' shards of the variant (~73 GB instead of 174 GB)
  2. Hard-link the base offload checkpoint's ple-store.json + ple-q4.rows
  3. Write an index without PLE keys and a config with `ple_storage` set

Prerequisite: the base offload checkpoint exists (scripts/build_ple_offload_checkpoint.py,
with materialization, i.e. ple-q4.rows present).

Run under the tool venv python (huggingface_hub installed; needs HF_TOKEN for gated repos):
  ~/.local/share/uv/tools/mlx-server/bin/python scripts/build_ple_overlay_checkpoint.py

  --weights-repo   variant repo (default orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX)
  --subdir         subfolder holding the quantization (default: repo root = 4-bit)
  --ple-dir        base offload checkpoint to take the PLE store from
  --dest-dir       output checkpoint dir
  --skip-download  reuse files already in --work-dir

The 4-bit variant has a different non-PLE quantization (group 64) than the base
(group 32); that is fine, config.json of the variant is authoritative for those
tensors. Do not mix config.json files between the two.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PLE_MARKER = ".ple.ple_embedding.ngram_embedding.shards."
CACHE_ROOT = Path.home() / ".cache" / "mlx-vlm"
DEFAULT_REPO = "orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX"
DEFAULT_PLE_DIR = CACHE_ROOT / "qwen3.8-flash-next-ple-offload"
DEFAULT_WORK_DIR = CACHE_ROOT / "qwen3.8-flash-next-uncensored-src"
DEFAULT_DEST_DIR = CACHE_ROOT / "qwen3.8-flash-next-uncensored-ple-offload"

# Small non-weight files copied verbatim from the variant.
AUX_FILES = [
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "README.md",
]


def step(title: str) -> None:
    print(f"\n{'-' * 60}\n{title}\n{'-' * 60}", flush=True)


def download(repo: str, subdir: str, work_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    prefix = f"{subdir}/" if subdir else ""

    def fetch(name: str) -> Path:
        return Path(
            hf_hub_download(repo, prefix + name, local_dir=work_dir, token=os.environ.get("HF_TOKEN"))
        )

    step(f"Download index + config from {repo}/{subdir or '.'}")
    index_path = fetch("model.safetensors.index.json")
    fetch("config.json")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    needed = sorted({f for k, f in weight_map.items() if PLE_MARKER not in k})
    skipped = sorted(set(weight_map.values()) - set(needed))
    print(f"{len(needed)} shards with non-PLE tensors; skipping {len(skipped)} pure-PLE shards")
    for i, name in enumerate(needed, 1):
        print(f"[{i}/{len(needed)}] {name}", flush=True)
        fetch(name)
    for name in AUX_FILES:
        try:
            fetch(name)
        except Exception as exc:  # optional files
            print(f"  (skip {name}: {type(exc).__name__})")


def assemble(repo: str, subdir: str, work_dir: Path, ple_dir: Path, dest_dir: Path) -> None:
    step("Assemble offload checkpoint")
    src = work_dir / subdir if subdir else work_dir
    for name in ("ple-store.json", "ple-q4.rows"):
        if not (ple_dir / name).exists():
            sys.exit(f"missing {ple_dir / name} - build the base offload checkpoint first")
    manifest = json.loads((ple_dir / "ple-store.json").read_text())
    if manifest.get("layout") != "interleaved_q4":
        sys.exit("ple-store.json is not a materialized (interleaved_q4) store")

    if dest_dir.exists() and any(dest_dir.iterdir()):
        sys.exit(f"dest dir not empty: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.stat().st_dev != dest_dir.stat().st_dev or ple_dir.stat().st_dev != dest_dir.stat().st_dev:
        sys.exit("work dir, ple dir and dest dir must share a filesystem for hard links")

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = {k: f for k, f in index["weight_map"].items() if PLE_MARKER not in k}
    ref_index = json.loads((ple_dir / "model.safetensors.index.json").read_text())["weight_map"]
    if set(weight_map) != set(ref_index):
        diff = set(weight_map) ^ set(ref_index)
        sys.exit(f"non-PLE key set differs from base offload checkpoint ({len(diff)} keys, e.g. {sorted(diff)[:3]})")

    used = sorted(set(weight_map.values()))
    for name in used:
        os.link(src / name, dest_dir / name)
    new_index = dict(index)
    new_index["weight_map"] = weight_map
    new_index.setdefault("metadata", {})["external_ple_bytes"] = json.loads(
        (ple_dir / "model.safetensors.index.json").read_text()
    ).get("metadata", {}).get("external_ple_bytes", "0")
    (dest_dir / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2) + "\n")

    for name in ("ple-store.json", "ple-q4.rows"):
        os.link(ple_dir / name, dest_dir / name)
    for name in AUX_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dest_dir / name)

    config = json.loads((src / "config.json").read_text())
    config["text_config"]["ple_storage"] = {"manifest": "ple-store.json", "cache_rows": 0}
    quant = config.get("quantization", {})
    config["quantization"] = {k: v for k, v in quant.items() if PLE_MARKER not in k}
    (dest_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    (dest_dir / "EXTERNAL_PLE.json").write_text(
        json.dumps(
            {
                "format": "qwen4_exp_external_ple_v1",
                "weights_repo": f"{repo}/{subdir}".rstrip("/"),
                "ple_store_from": str(ple_dir),
                "external_ple_manifest": "ple-store.json",
                "linked_weight_files": used,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"done: {dest_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-repo", default=DEFAULT_REPO)
    ap.add_argument("--subdir", default="")
    ap.add_argument("--ple-dir", type=Path, default=DEFAULT_PLE_DIR)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--dest-dir", type=Path, default=DEFAULT_DEST_DIR)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    if not a.skip_download:
        download(a.weights_repo, a.subdir, a.work_dir)
    assemble(a.weights_repo, a.subdir, a.work_dir, a.ple_dir, a.dest_dir)


if __name__ == "__main__":
    main()
