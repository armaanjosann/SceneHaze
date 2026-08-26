"""
Build a 4-panel comparison figure (clean | constant-beta | scene-grounded |
beta map) plus an amplified diff image (grounded minus constant), for one
image where clean/constant/grounded/betamap outputs already exist.

Usage:
  python3 scripts/make_comparison.py --split train --city strasbourg \
      --image strasbourg_000001_043748 --beta 2.0
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from fog_utils import PROJECT_ROOT, CLEAN_ROOT, load_and_resize, label_panel, make_grid

CONSTANT_ROOT = PROJECT_ROOT / "data" / "generated" / "constant_beta"
GROUNDED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded"
OUT_ROOT = PROJECT_ROOT / "results" / "comparisons"


def make_diff(constant_path: Path, grounded_path: Path, out_path: Path, amplify: float = 4.0):
    const = np.array(Image.open(constant_path).convert("RGB")).astype(np.float32)
    grounded = np.array(Image.open(grounded_path).convert("RGB")).astype(np.float32)
    diff = grounded - const  # positive = grounded is foggier here, negative = grounded is clearer
    vis = np.clip(128 + diff.mean(axis=2) * amplify, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(out_path)
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", required=True)
    ap.add_argument("--image", required=True, help="image stem, no suffix")
    ap.add_argument("--beta", type=float, default=2.0, help="beta / beta_base value used for generation")
    args = ap.parse_args()

    clean_path = CLEAN_ROOT / args.split / args.city / f"{args.image}_leftImg8bit.png"
    constant_path = CONSTANT_ROOT / args.split / args.city / f"{args.image}_beta{args.beta:.2f}_constant.png"
    grounded_stem = f"{args.image}_betabase{args.beta:.2f}_grounded"
    grounded_path = GROUNDED_ROOT / args.split / args.city / f"{grounded_stem}.png"
    betamap_path = GROUNDED_ROOT / args.split / args.city / f"{grounded_stem}_betamap.png"

    panels = [
        (clean_path, "clean (original)"),
        (constant_path, f"constant beta={args.beta:.1f} (baseline)"),
        (grounded_path, f"scene-grounded beta_base={args.beta:.1f} (ours)"),
        (betamap_path, "beta map (dark=sky x0.3, bright=road x1.3)"),
    ]

    labeled = [label_panel(load_and_resize(p), text) for p, text in panels]
    grid = make_grid(labeled, cols=2)

    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / f"{args.city}_{args.image}_beta{args.beta:.1f}_comparison.png"
    grid.save(grid_path)
    print(f"Saved comparison grid: {grid_path}")

    diff_path = out_dir / f"{args.city}_{args.image}_beta{args.beta:.1f}_diff.png"
    diff = make_diff(constant_path, grounded_path, diff_path)
    print(f"Saved diff image: {diff_path}  (raw diff range [{diff.min():.1f}, {diff.max():.1f}])")


if __name__ == "__main__":
    main()
