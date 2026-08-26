"""
Piece 1 — constant-beta ASM fog generation (baseline 1 / RESIDE-style).

I(x) = J(x) * t(x) + A * (1 - t(x)), t(x) = exp(-beta * d(x))

beta is a single scalar: fog density is uniform in "distance space" — it
only varies across the image because depth varies, not because the scene
does. This is the baseline the scene-grounded method (Piece 2) is meant to
beat.

Usage:
  python3 scripts/generate_constant.py \
      --image aachen_000000_000019 --split train --city aachen --beta 1.0
"""

import argparse
from pathlib import Path

from fog_utils import (
    PROJECT_ROOT,
    apply_asm,
    disparity_to_pseudo_depth,
    load_clean,
    load_disparity,
    save_image,
)

OUT_ROOT = PROJECT_ROOT / "data" / "generated" / "constant_beta"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="aachen_000000_000019", help="image stem, no suffix")
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", default="aachen")
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    clean = load_clean(args.split, args.city, args.image)
    disparity = load_disparity(args.split, args.city, args.image)
    depth = disparity_to_pseudo_depth(disparity)

    foggy = apply_asm(clean, depth, args.beta)

    out_path = OUT_ROOT / args.split / args.city / f"{args.image}_beta{args.beta:.2f}_constant.png"
    save_image(foggy, out_path)

    print(f"beta = {args.beta}")
    print(f"pseudo-depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
