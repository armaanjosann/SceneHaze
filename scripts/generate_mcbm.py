"""
Piece 3 — MCBM fog generation (baseline 2), adapted from HazeFlow
(Shin et al., ICCV 2025): github.com/cloor/HazeFlow

beta(x) = beta_base + nh(x) * scale

where nh(x) is one of HazeFlow's 1000 precomputed 256x256 Brownian-motion
random fields (external/HazeFlow/datasets/MCBM/*.png), upsampled to image
resolution, and scale ~ Uniform(0.5, 1.0). This is genuinely spatially
varying fog, but the variation is pure noise — it has no relationship to
scene content (depth aside, which every baseline shares).

Fidelity note: this reproduces HazeFlow's own beta formula, taken directly
from their training dataloader (reflow/datasets.py::MCBM.__getitem__).
Their loader plugs it into t = exp(-depth * 2.0 * beta) at a fixed 256x256
training-crop resolution. For a fair three-way comparison against our
constant-beta and scene-grounded methods — same clean image, same depth
map, only beta(x) computation differs (see project context doc) — we keep
our own equation form t = exp(-beta(x)*depth) via fog_utils.apply_asm, and
only borrow HazeFlow's actual random-field generator/assets for the
spatial noise pattern. We also skip HazeFlow's requirements.txt (jax,
tensorflow, cuda-only torch) entirely, since only the static PNG fields are
needed here.

Usage:
  python3 scripts/generate_mcbm.py \
      --image strasbourg_000001_043748 --split train --city strasbourg \
      --beta-base 2.0 --mcbm-index 42 --scale 0.75
"""

import argparse
import random

import numpy as np
from PIL import Image

from fog_utils import (
    PROJECT_ROOT,
    apply_asm,
    disparity_to_pseudo_depth,
    load_clean,
    load_disparity,
    save_image,
)

OUT_ROOT = PROJECT_ROOT / "data" / "generated" / "mcbm"
MCBM_DIR = PROJECT_ROOT / "external" / "HazeFlow" / "datasets" / "MCBM"

# Matches (np.random.rand()+1)/2 in HazeFlow's own dataloader -> [0.5, 1.0]
SCALE_RANGE = (0.5, 1.0)


def load_mcbm_map(index: int, shape) -> np.ndarray:
    path = MCBM_DIR / f"{index}.png"
    if not path.exists():
        raise SystemExit(f"Missing MCBM map: {path} (expected index 0-999)")
    nh = Image.open(path).convert("L").resize((shape[1], shape[0]), Image.BICUBIC)
    nh = np.array(nh).astype(np.float32) / 255.0
    return (nh - nh.min()) / (nh.max() - nh.min() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="aachen_000000_000019", help="image stem, no suffix")
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", default="aachen")
    ap.add_argument("--beta-base", type=float, default=1.0)
    ap.add_argument("--mcbm-index", type=int, default=None, help="0-999; random if omitted")
    ap.add_argument("--scale", type=float, default=None, help="nh contribution scale; random in [0.5,1.0] if omitted (matches HazeFlow)")
    ap.add_argument("--seed", type=int, default=None, help="seed for mcbm-index/scale randomness, if not explicitly set")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    clean = load_clean(args.split, args.city, args.image)
    disparity = load_disparity(args.split, args.city, args.image)
    depth = disparity_to_pseudo_depth(disparity)

    mcbm_index = args.mcbm_index if args.mcbm_index is not None else rng.randint(0, 999)
    scale = args.scale if args.scale is not None else rng.uniform(*SCALE_RANGE)

    nh = load_mcbm_map(mcbm_index, depth.shape)
    beta_map = args.beta_base + nh * scale

    foggy = apply_asm(clean, depth, beta_map)

    out_dir = OUT_ROOT / args.split / args.city
    stem = f"{args.image}_betabase{args.beta_base:.2f}_mcbm{mcbm_index}"
    save_image(foggy, out_dir / f"{stem}.png")

    beta_vis = (beta_map - beta_map.min()) / (beta_map.max() - beta_map.min() + 1e-8)
    save_image(np.repeat(beta_vis[:, :, None], 3, axis=2), out_dir / f"{stem}_betamap.png")

    print(f"mcbm_index={mcbm_index}, scale={scale:.3f}, beta_base={args.beta_base}")
    print(f"beta_map range: [{beta_map.min():.3f}, {beta_map.max():.3f}]")
    print(f"Saved: {out_dir / (stem + '.png')}")
    print(f"Saved beta map viz: {out_dir / (stem + '_betamap.png')}")


if __name__ == "__main__":
    main()
