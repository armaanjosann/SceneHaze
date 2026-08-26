"""
Piece 4 (partial) — three-way comparison: constant-beta (baseline 1) vs
MCBM (baseline 2, HazeFlow) vs scene-grounded (ours), all from the same
clean image + same depth map, differing only in how beta(x) is computed.

Runs all three generators in one process (so the randomly-chosen MCBM index
doesn't need to be re-discovered from filenames) and produces:
  - <city>_<image>_threeway_beta{B}.png   : clean | constant | mcbm | grounded
  - <city>_<image>_betamaps_beta{B}.png   : grounded beta map | mcbm beta map
                                            (informed structure vs random noise)

Usage:
  python3 scripts/run_threeway.py --split train --city strasbourg \
      --image strasbourg_000001_043748 --beta-base 2.0 --seed 0
"""

import argparse
import random

import numpy as np
from PIL import Image

from fog_utils import (
    PROJECT_ROOT,
    CLEAN_ROOT,
    apply_asm,
    disparity_to_pseudo_depth,
    load_clean,
    load_disparity,
    load_seg_labels,
    save_image,
    load_and_resize,
    label_panel,
    make_grid,
)
from generate_grounded import build_beta_map as build_grounded_beta_map
from generate_mcbm import load_mcbm_map, SCALE_RANGE

CONSTANT_ROOT = PROJECT_ROOT / "data" / "generated" / "constant_beta"
GROUNDED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded"
MCBM_ROOT = PROJECT_ROOT / "data" / "generated" / "mcbm"
OUT_ROOT = PROJECT_ROOT / "results" / "comparisons"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", required=True)
    ap.add_argument("--image", required=True, help="image stem, no suffix")
    ap.add_argument("--beta-base", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=20.0, help="gaussian smoothing for grounded beta map (FID-optimized default)")
    ap.add_argument("--mcbm-index", type=int, default=None, help="0-999; random (seeded) if omitted")
    ap.add_argument("--scale", type=float, default=None, help="mcbm nh scale; random (seeded) in [0.5,1.0] if omitted")
    ap.add_argument("--seed", type=int, default=0, help="seed for mcbm-index/scale randomness")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    B = args.beta_base

    clean = load_clean(args.split, args.city, args.image)
    disparity = load_disparity(args.split, args.city, args.image)
    depth = disparity_to_pseudo_depth(disparity)
    seg = load_seg_labels(args.split, args.city, args.image)

    # --- baseline 1: constant beta ---
    constant_foggy = apply_asm(clean, depth, B)
    constant_path = CONSTANT_ROOT / args.split / args.city / f"{args.image}_beta{B:.2f}_constant.png"
    save_image(constant_foggy, constant_path)

    # --- baseline 2: MCBM (HazeFlow) ---
    mcbm_index = args.mcbm_index if args.mcbm_index is not None else rng.randint(0, 999)
    scale = args.scale if args.scale is not None else rng.uniform(*SCALE_RANGE)
    nh = load_mcbm_map(mcbm_index, depth.shape)
    mcbm_beta_map = B + nh * scale
    mcbm_foggy = apply_asm(clean, depth, mcbm_beta_map)
    mcbm_stem = f"{args.image}_betabase{B:.2f}_mcbm{mcbm_index}"
    mcbm_path = MCBM_ROOT / args.split / args.city / f"{mcbm_stem}.png"
    save_image(mcbm_foggy, mcbm_path)
    mcbm_betamap_vis = (mcbm_beta_map - mcbm_beta_map.min()) / (mcbm_beta_map.max() - mcbm_beta_map.min() + 1e-8)
    mcbm_betamap_path = MCBM_ROOT / args.split / args.city / f"{mcbm_stem}_betamap.png"
    save_image(np.repeat(mcbm_betamap_vis[:, :, None], 3, axis=2), mcbm_betamap_path)

    # --- ours: scene-grounded ---
    grounded_beta_map = build_grounded_beta_map(seg, depth, B, args.sigma)
    grounded_foggy = apply_asm(clean, depth, grounded_beta_map)
    grounded_stem = f"{args.image}_betabase{B:.2f}_grounded"
    grounded_path = GROUNDED_ROOT / args.split / args.city / f"{grounded_stem}.png"
    save_image(grounded_foggy, grounded_path)
    grounded_betamap_vis = (grounded_beta_map - grounded_beta_map.min()) / (grounded_beta_map.max() - grounded_beta_map.min() + 1e-8)
    grounded_betamap_path = GROUNDED_ROOT / args.split / args.city / f"{grounded_stem}_betamap.png"
    save_image(np.repeat(grounded_betamap_vis[:, :, None], 3, axis=2), grounded_betamap_path)

    clean_path = CLEAN_ROOT / args.split / args.city / f"{args.image}_leftImg8bit.png"

    # --- figure 1: clean | constant | mcbm | grounded ---
    panels = [
        (clean_path, "clean (original)"),
        (constant_path, f"constant beta={B:.1f} (baseline 1: uniform)"),
        (mcbm_path, f"MCBM beta_base={B:.1f} (baseline 2: random)"),
        (grounded_path, f"scene-grounded beta_base={B:.1f} (ours: scene-aware)"),
    ]
    labeled = [label_panel(load_and_resize(p), text) for p, text in panels]
    grid = make_grid(labeled, cols=2)
    grid_path = OUT_ROOT / f"{args.city}_{args.image}_beta{B:.1f}_threeway.png"
    grid.save(grid_path)

    # --- figure 2: grounded beta map | mcbm beta map (structure vs noise) ---
    betamap_panels = [
        (grounded_betamap_path, "scene-grounded beta map (structured: sky/road/veg)"),
        (mcbm_betamap_path, "MCBM beta map (random Brownian-motion field)"),
    ]
    labeled_bm = [label_panel(load_and_resize(p), text) for p, text in betamap_panels]
    bm_grid = make_grid(labeled_bm, cols=2)
    bm_grid_path = OUT_ROOT / f"{args.city}_{args.image}_beta{B:.1f}_betamap_compare.png"
    bm_grid.save(bm_grid_path)

    print(f"MCBM: index={mcbm_index}, scale={scale:.3f}")
    print(f"Saved threeway grid: {grid_path}")
    print(f"Saved beta-map comparison: {bm_grid_path}")


if __name__ == "__main__":
    main()
