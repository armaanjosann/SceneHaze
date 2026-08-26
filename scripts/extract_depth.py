"""
Run Depth Anything V2 on a handful of Cityscapes images as a sanity check.

Saves, per image, into data/depth/cityscapes/<split>/<city>/:
  - <name>_depth.npy  -> raw float32 depth (relative, NOT normalized) for downstream math
  - <name>_depth.png  -> normalized 0-255 grayscale visualization, for eyeballing

Usage:
  python3 scripts/extract_depth.py                # default: 5 images from train/aachen
  python3 scripts/extract_depth.py --n 3 --city bochum
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_ROOT = PROJECT_ROOT / "data" / "clean" / "cityscapes"
DEPTH_ROOT = PROJECT_ROOT / "data" / "depth" / "cityscapes"

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", default="aachen")
    ap.add_argument("--n", type=int, default=5, help="number of images to process (ignored if --images given)")
    ap.add_argument("--images", nargs="+", default=None, help="specific image stems (no suffix) to process")
    args = ap.parse_args()

    src_dir = CLEAN_ROOT / args.split / args.city
    if not src_dir.exists():
        raise SystemExit(f"No such source dir: {src_dir}")

    if args.images:
        images = [src_dir / f"{stem}_leftImg8bit.png" for stem in args.images]
        missing = [p for p in images if not p.exists()]
        if missing:
            raise SystemExit(f"Missing source image(s): {missing}")
    else:
        images = sorted(src_dir.glob("*_leftImg8bit.png"))[: args.n]
    if not images:
        raise SystemExit(f"No images found in {src_dir}")

    dst_dir = DEPTH_ROOT / args.split / args.city
    dst_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading {MODEL_ID} ...")
    depth_pipe = pipeline(task="depth-estimation", model=MODEL_ID, device=device)

    for img_path in images:
        name = img_path.stem.replace("_leftImg8bit", "")
        print(f"  {name} ...", end=" ", flush=True)

        image = Image.open(img_path).convert("RGB")
        result = depth_pipe(image)

        # HF depth-estimation pipeline returns a "predicted_depth" tensor
        # (relative depth, not metric) and a "depth" PIL preview image.
        depth_tensor = result["predicted_depth"]
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.squeeze(0)
        depth_np = depth_tensor.detach().cpu().numpy().astype(np.float32)

        np.save(dst_dir / f"{name}_depth.npy", depth_np)

        # normalized visualization (near=bright or dark depending on model convention;
        # Depth Anything outputs *disparity-like* values: larger = closer)
        d = depth_np
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        vis = (d_norm * 255).astype(np.uint8)
        Image.fromarray(vis).save(dst_dir / f"{name}_depth.png")

        print(f"depth range [{d.min():.3f}, {d.max():.3f}] -> saved")

    print(f"\nDone. {len(images)} depth maps written to {dst_dir}")


if __name__ == "__main__":
    main()
