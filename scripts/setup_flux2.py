#!/usr/bin/env python3
"""
Setup script for FLUX.2-klein models.

What this does:
  1. Provisions the image venv (if not already done)
  2. Pre-downloads and quantizes models via mflux-save (avoids slow HF download on cold start)
  3. Downloads ponpoke/flux2-klein-9b-uncensored-text-encoder (abliterated T5 encoder)
  4. Merges the abliterated encoder into the pre-cached 9B model snapshot
  5. Prints Civitai LoRA download instructions

After running this script, update config/models.yaml as instructed in the output.

Usage:
  python scripts/setup_flux2.py                # full setup
  python scripts/setup_flux2.py --skip-models  # encoder + LoRA instructions only
  python scripts/setup_flux2.py --skip-encoder # model pre-cache only
  python scripts/setup_flux2.py --4b-only      # only pre-cache the 4B fast model
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "mflux"
LORA_DIR = CACHE_DIR / "loras"
ENCODER_DIR = CACHE_DIR / "encoders"
MODEL_4B_DIR = CACHE_DIR / "flux2-klein-4b-8bit"
MODEL_9B_DIR = CACHE_DIR / "flux2-klein-9b-8bit"
MODEL_9B_UNCENSORED_DIR = CACHE_DIR / "flux2-klein-9b-8bit-uncensored"
VENV_IMAGE = Path.home() / ".local" / "share" / "mlx" / "venvs" / "image"


def run(args: list, **kw) -> None:
    print(f"  $ {' '.join(str(a) for a in args)}", flush=True)
    subprocess.run([str(a) for a in args], check=True, **kw)


def step(title: str) -> None:
    print(f"\n{'─' * 60}\n{title}\n{'─' * 60}")


def provision_venv() -> None:
    step("Step 1: Image venv")
    if VENV_IMAGE.exists():
        print(f"✓ Image venv already provisioned at {VENV_IMAGE}")
        return
    print("Provisioning image venv via `mlx install image` …")
    mlx_bin = shutil.which("mlx")
    if mlx_bin:
        run([mlx_bin, "install", "image"])
    else:
        # Fallback: run via uv tool
        run([sys.executable, "-m", "mlx_server.cli", "install", "image"])


def _mflux_save_bin() -> Path:
    bin_path = VENV_IMAGE / "bin" / "mflux-save"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"mflux-save not found at {bin_path}.\n"
            "Run `mlx install image` first (or run this script without --skip-models to auto-provision)."
        )
    return bin_path


def precache_models(only_4b: bool = False) -> None:
    step("Step 2: Pre-cache models (mflux-save)")
    save = _mflux_save_bin()

    if not MODEL_4B_DIR.exists():
        print(f"Downloading + quantizing flux2-klein-4b (8-bit) → {MODEL_4B_DIR}")
        print("  This downloads ~15 GB from HuggingFace and may take a while.")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        run([save, "--model", "flux2-klein-4b", "--quantize", "8", "--save-path", str(MODEL_4B_DIR)])
        print(f"✓ 4B model cached at {MODEL_4B_DIR}")
    else:
        print(f"✓ 4B model already cached at {MODEL_4B_DIR}")

    if only_4b:
        print("Skipping 9B model (--4b-only).")
        return

    if not MODEL_9B_DIR.exists():
        print(f"Downloading + quantizing flux2-klein-9b (8-bit) → {MODEL_9B_DIR}")
        print("  This downloads ~32 GB from HuggingFace and may take a while.")
        run([save, "--model", "flux2-klein-9b", "--quantize", "8", "--save-path", str(MODEL_9B_DIR)])
        print(f"✓ 9B model cached at {MODEL_9B_DIR}")
    else:
        print(f"✓ 9B model already cached at {MODEL_9B_DIR}")


def download_and_merge_encoder(only_4b: bool = False) -> None:
    step("Step 3: Abliterated text encoder (ponpoke/flux2-klein-9b-uncensored-text-encoder)")

    if only_4b:
        print("Skipping encoder (--4b-only; encoder is 9B-specific).")
        return

    encoder_local = ENCODER_DIR / "flux2-klein-9b-uncensored-t5"

    if not encoder_local.exists():
        print(f"Downloading ponpoke/flux2-klein-9b-uncensored-text-encoder → {encoder_local}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("  huggingface_hub not installed in current env.")
            print("  Install with: pip install huggingface_hub")
            print("  Or: ~/.local/share/uv/tools/mlx-server/bin/pip install huggingface_hub")
            return
        ENCODER_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="ponpoke/flux2-klein-9b-uncensored-text-encoder",
            local_dir=str(encoder_local),
        )
        print(f"✓ Encoder downloaded to {encoder_local}")
    else:
        print(f"✓ Encoder already at {encoder_local}")

    _merge_encoder(encoder_local)


def _merge_encoder(encoder_dir: Path) -> None:
    if MODEL_9B_UNCENSORED_DIR.exists():
        print(f"✓ Merged uncensored model already at {MODEL_9B_UNCENSORED_DIR}")
        return

    if not MODEL_9B_DIR.exists():
        print(
            f"⚠ 9B model not yet pre-cached at {MODEL_9B_DIR}.\n"
            "  Run without --skip-models to pre-cache it first, then re-run with --skip-models."
        )
        return

    print(f"Merging abliterated T5 encoder into 9B snapshot → {MODEL_9B_UNCENSORED_DIR}")
    shutil.copytree(str(MODEL_9B_DIR), str(MODEL_9B_UNCENSORED_DIR))

    # FLUX.2 uses CLIP (text_encoder) + T5 (text_encoder_2).
    # ponpoke's repo replaces text_encoder_2 with the abliterated version.
    src_t5 = encoder_dir / "text_encoder_2"
    dst_t5 = MODEL_9B_UNCENSORED_DIR / "text_encoder_2"

    if src_t5.exists():
        if dst_t5.exists():
            shutil.rmtree(str(dst_t5))
        shutil.copytree(str(src_t5), str(dst_t5))
        print(f"  ✓ text_encoder_2 replaced with abliterated version.")
    else:
        # Fallback: root-level safetensors (some repos lay flat)
        encoder_files = list(encoder_dir.glob("*.safetensors"))
        if encoder_files:
            dst_t5.mkdir(parents=True, exist_ok=True)
            for f in encoder_files:
                shutil.copy2(str(f), str(dst_t5 / f.name))
            print(f"  ✓ Copied {len(encoder_files)} encoder file(s) into text_encoder_2/.")
        else:
            shutil.rmtree(str(MODEL_9B_UNCENSORED_DIR))
            print(
                f"  ⚠ Could not identify encoder files in {encoder_dir}.\n"
                "    Expected a text_encoder_2/ subdirectory or root-level .safetensors files.\n"
                "    Check the repo structure and merge manually."
            )
            return

    print(f"✓ Uncensored 9B model ready at {MODEL_9B_UNCENSORED_DIR}")


def print_lora_instructions() -> None:
    step("Step 4: NSFW LoRA — Manual Download (Civitai account required)")
    print(f"""
For visual NSFW content, you need a Klein-native NSFW LoRA.
(FLUX.1 LoRAs like aidmaNSFWunlock are NOT compatible — different architecture.)

┌─────────────────────────────────────────────────────────────┐
│ Recommended LoRA for FLUX.2-klein-9B:                       │
│                                                             │
│ "Ultra Real – Klein 9b KL_9B_V4"                           │
│  URL:  https://civitai.com/models/2462105                   │
│  Size: ~158 MB (.safetensors)                               │
│  Save: {str(LORA_DIR / 'ultra-real-klein-v4.safetensors'):<51}│
└─────────────────────────────────────────────────────────────┘

For FLUX.2-klein-4B LoRAs (9B LoRA weight shapes won't load into 4B):
  https://civitai.com/search/models?query=flux2+klein+4b+lora

After downloading, update models.yaml for flux2-klein-hifi:

  extra_args:
    - "--quantize"
    - "8"
    - "--lora-paths"
    - "~/.cache/mflux/loras/ultra-real-klein-v4.safetensors"
    - "--lora-scales"
    - "0.8"
""")


def print_yaml_instructions(only_4b: bool) -> None:
    step("models.yaml — Optional: Use pre-cached paths for faster startup")
    lines = []

    if MODEL_4B_DIR.exists():
        lines.append(f"  flux2-klein-fast  →  source: {MODEL_4B_DIR}")

    if not only_4b:
        if MODEL_9B_UNCENSORED_DIR.exists():
            lines.append(
                f"  flux2-klein-hifi  →  source: {MODEL_9B_UNCENSORED_DIR}\n"
                f"                         extra_args: add \"--model-path\", \"{MODEL_9B_UNCENSORED_DIR}\""
            )
        elif MODEL_9B_DIR.exists():
            lines.append(f"  flux2-klein-hifi  →  source: {MODEL_9B_DIR}")

    if lines:
        print("Update config/models.yaml source: fields to skip HF download on cold start:\n")
        for line in lines:
            print(f"  {line}")
    else:
        print("  (no pre-cached models found; models will download from HuggingFace on first start)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--skip-models", action="store_true", help="Skip mflux-save pre-cache steps")
    p.add_argument("--skip-encoder", action="store_true", help="Skip abliterated text encoder download")
    p.add_argument("--4b-only", dest="only_4b", action="store_true", help="Only set up the 4B fast model")
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)

    print("=== FLUX.2-klein Setup ===")

    provision_venv()

    if not args.skip_models:
        precache_models(only_4b=args.only_4b)
    else:
        print("\nSkipping model pre-cache (--skip-models).")

    if not args.skip_encoder:
        download_and_merge_encoder(only_4b=args.only_4b)
    else:
        print("\nSkipping text encoder (--skip-encoder).")

    print_lora_instructions()
    print_yaml_instructions(only_4b=args.only_4b)

    print("\n=== Done ===")
    print("Next steps:")
    print("  1. Download LoRA from Civitai (see above)")
    print("  2. Update config/models.yaml with LoRA path (and optional pre-cached source)")
    print("  3. mlx start --model flux2-klein-fast")
    print("  4. mlx start --model flux2-klein-hifi")


if __name__ == "__main__":
    main()
