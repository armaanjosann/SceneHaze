"""
Build the full three-way comparison figure: clean | constant-beta | MCBM |
scene-grounded | grounded beta map | MCBM beta map (6 panels), plus amplified
diff images (grounded vs constant, grounded vs MCBM).

Assumes constant/mcbm/grounded outputs already exist for this image+beta
(generate them first via generate_constant.py, generate_mcbm.py,
generate_grounded.py — or run_threeway.py, which does all three at once).

Usage:
  python3 scripts/make_comparison.py --split train --city strasbourg \
      --image strasbourg_000001_043748 --beta 2.0

  # if more than one MCBM output exists for this image+beta (different
  # random draws), disambiguate with:
  python3 scripts/make_comparison.py --split train --city strasbourg \
      --image strasbourg_000001_043748 --beta 2.0 --mcbm-index 864
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from fog_utils import PROJECT_ROOT, CLEAN_ROOT, load_and_resize, label_panel, make_grid

CONSTANT_ROOT = PROJECT_ROOT / "data" / "generated" / "constant_beta"
GROUNDED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded"
MCBM_ROOT = PROJECT_ROOT / "data" / "generated" / "mcbm"
OUT_ROOT = PROJECT_ROOT / "results" / "comparisons"


def find_mcbm_path(split: str, city: str, image: str, beta: float, mcbm_index: int = None) -> Path:
    mcbm_dir = MCBM_ROOT / split / city
    if mcbm_index is not None:
        path = mcbm_dir / f"{image}_betabase{beta:.2f}_mcbm{mcbm_index}.png"
        if not path.exists():
            raise SystemExit(f"Missing MCBM file: {path}")
        return path

    matches = sorted(mcbm_dir.glob(f"{image}_betabase{beta:.2f}_mcbm*.png"))
    matches = [p for p in matches if not p.stem.endswith("_betamap")]
    if not matches:
        raise SystemExit(
            f"No MCBM output found for {image} at beta={beta:.2f} in {mcbm_dir}\n"
            f"Run: python3 scripts/generate_mcbm.py --split {split} --city {city} "
            f"--image {image} --beta-base {beta}"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"Multiple MCBM outputs found for {image} at beta={beta:.2f}: "
            f"{[p.name for p in matches]}\nPass --mcbm-index to disambiguate."
        )
    return matches[0]


def make_diff(a_path: Path, b_path: Path, out_path: Path, amplify: float = 4.0):
    """diff = b - a. Positive = b is foggier here, negative = b is clearer."""
    a = np.array(Image.open(a_path).convert("RGB")).astype(np.float32)
    b = np.array(Image.open(b_path).convert("RGB")).astype(np.float32)
    diff = b - a
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
    ap.add_argument("--mcbm-index", type=int, default=None, help="disambiguate if multiple MCBM draws exist for this image+beta")
    args = ap.parse_args()

    B = args.beta
    clean_path = CLEAN_ROOT / args.split / args.city / f"{args.image}_leftImg8bit.png"
    constant_path = CONSTANT_ROOT / args.split / args.city / f"{args.image}_beta{B:.2f}_constant.png"
    grounded_stem = f"{args.image}_betabase{B:.2f}_grounded"
    grounded_path = GROUNDED_ROOT / args.split / args.city / f"{grounded_stem}.png"
    grounded_betamap_path = GROUNDED_ROOT / args.split / args.city / f"{grounded_stem}_betamap.png"

    mcbm_path = find_mcbm_path(args.split, args.city, args.image, B, args.mcbm_index)
    mcbm_betamap_path = mcbm_path.with_name(mcbm_path.stem + "_betamap.png")

    panels = [
        (clean_path, "clean (original)"),
        (constant_path, f"constant beta={B:.1f} (baseline 1: uniform)"),
        (mcbm_path, f"MCBM beta_base={B:.1f} (baseline 2: random)"),
        (grounded_path, f"scene-grounded beta_base={B:.1f} (ours: scene-aware)"),
        (grounded_betamap_path, "scene-grounded beta map (structured)"),
        (mcbm_betamap_path, "MCBM beta map (random field)"),
    ]

    labeled = [label_panel(load_and_resize(p), text) for p, text in panels]
    grid = make_grid(labeled, cols=3)

    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / f"{args.city}_{args.image}_beta{B:.1f}_comparison.png"
    grid.save(grid_path)
    print(f"Saved comparison grid: {grid_path}")

    diff_vs_constant_path = out_dir / f"{args.city}_{args.image}_beta{B:.1f}_diff_vs_constant.png"
    diff1 = make_diff(constant_path, grounded_path, diff_vs_constant_path)
    print(f"Saved diff (grounded vs constant): {diff_vs_constant_path}  (range [{diff1.min():.1f}, {diff1.max():.1f}])")

    diff_vs_mcbm_path = out_dir / f"{args.city}_{args.image}_beta{B:.1f}_diff_vs_mcbm.png"
    diff2 = make_diff(mcbm_path, grounded_path, diff_vs_mcbm_path)
    print(f"Saved diff (grounded vs MCBM): {diff_vs_mcbm_path}  (range [{diff2.min():.1f}, {diff2.max():.1f}])")


if __name__ == "__main__":
    main()
