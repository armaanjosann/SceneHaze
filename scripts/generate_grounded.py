"""
Piece 2 — scene-grounded ASM fog generation (our method).

Same ASM equation as Piece 1, but beta becomes a per-pixel map instead of a
scalar: beta(x) = beta_base * category_modifier(x), where the modifier comes
from Cityscapes ground-truth semantic labels (_gtFine_labelIds.png).

  Sky              (23)              x0.3   fog thins above the boundary layer
  Ground/road      (7,8,9,10)        x1.3   fog pools at ground level
  Low vegetation   (21,22)           x1.1   ground-level, traps moisture
  Everything else  (incl. structures) x1.0  default — no change from baseline

("Water" has no Cityscapes label — it's added later via SegFormer/ADE20K
when we move beyond Cityscapes.)

The per-pixel beta map is then Gaussian-smoothed (sigma=15 by default) so
category boundaries don't create harsh fog edges — real fog doesn't respect
segmentation masks.

Optional --turbulence layers small-scale organic variation on top of that
large-scale structure: the beta map is multiplied by smooth noise centered
at 1.0 (see fog_utils.apply_turbulence), so pockets of slightly thicker or
thinner fog appear within a single terrain type, without disturbing the
sky/ground/vegetation structure itself.

Usage:
  python3 scripts/generate_grounded.py \
      --image aachen_000000_000019 --split train --city aachen --beta-base 1.0

  python3 scripts/generate_grounded.py \
      --image aachen_000000_000019 --split train --city aachen --beta-base 1.0 \
      --turbulence --turb-strength 0.15 --turb-scale 20 --turb-seed 0
"""

import argparse

import numpy as np
from scipy.ndimage import gaussian_filter

from fog_utils import (
    PROJECT_ROOT,
    apply_asm,
    apply_turbulence,
    disparity_to_pseudo_depth,
    load_clean,
    load_disparity,
    load_seg_labels,
    save_image,
)

OUT_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded"

# Cityscapes labelId -> fog-category modifier (see project context doc)
SKY_IDS = {23}
GROUND_IDS = {7, 8, 9, 10}       # road, sidewalk, parking, rail track
VEGETATION_IDS = {21, 22}        # vegetation, terrain
# structures (11-16) and everything else default to x1.0 -> no-op, omitted

SKY_MODIFIER = 0.3
GROUND_MODIFIER = 1.3
VEGETATION_MODIFIER = 1.1


def build_beta_map(seg: np.ndarray, beta_base: float, sigma: float) -> np.ndarray:
    beta_map = np.full(seg.shape, beta_base, dtype=np.float32)
    beta_map[np.isin(seg, list(SKY_IDS))] *= SKY_MODIFIER
    beta_map[np.isin(seg, list(GROUND_IDS))] *= GROUND_MODIFIER
    beta_map[np.isin(seg, list(VEGETATION_IDS))] *= VEGETATION_MODIFIER
    return gaussian_filter(beta_map, sigma=sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="aachen_000000_000019", help="image stem, no suffix")
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", default="aachen")
    ap.add_argument("--beta-base", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=15.0, help="gaussian smoothing of beta map")
    ap.add_argument("--turbulence", action="store_true", help="layer small-scale organic noise on top of the scene-grounded beta map")
    ap.add_argument("--turb-strength", type=float, default=0.15, help="std dev of turbulence noise, centered at 1.0")
    ap.add_argument("--turb-scale", type=float, default=20.0, help="gaussian sigma smoothing the turbulence noise field (bigger = softer/larger blobs)")
    ap.add_argument("--turb-seed", type=int, default=None, help="seed for turbulence noise (reproducibility)")
    args = ap.parse_args()

    clean = load_clean(args.split, args.city, args.image)
    disparity = load_disparity(args.split, args.city, args.image)
    depth = disparity_to_pseudo_depth(disparity)
    seg = load_seg_labels(args.split, args.city, args.image)

    if seg.shape != depth.shape:
        raise SystemExit(
            f"Shape mismatch: seg {seg.shape} vs depth {depth.shape} — "
            f"clean/seg/depth must all be the same resolution."
        )

    beta_map = build_beta_map(seg, args.beta_base, args.sigma)

    stem = f"{args.image}_betabase{args.beta_base:.2f}_grounded"
    if args.turbulence:
        beta_map = apply_turbulence(beta_map, args.turb_strength, args.turb_scale, args.turb_seed)
        stem += f"_turb{args.turb_strength:.2f}-{args.turb_scale:.0f}"

    foggy = apply_asm(clean, depth, beta_map)

    out_dir = OUT_ROOT / args.split / args.city
    save_image(foggy, out_dir / f"{stem}.png")

    # Also save a visualization of the beta map itself — useful for sanity
    # checking that the modifiers are landing where expected, and for
    # thesis figures later.
    beta_vis = (beta_map - beta_map.min()) / (beta_map.max() - beta_map.min() + 1e-8)
    save_image(np.repeat(beta_vis[:, :, None], 3, axis=2), out_dir / f"{stem}_betamap.png")

    print(f"beta_base = {args.beta_base}, sigma = {args.sigma}")
    if args.turbulence:
        print(f"turbulence: strength={args.turb_strength}, scale={args.turb_scale}, seed={args.turb_seed}")
    print(f"beta_map range: [{beta_map.min():.3f}, {beta_map.max():.3f}]")
    print(f"Saved: {out_dir / (stem + '.png')}")
    print(f"Saved beta map viz: {out_dir / (stem + '_betamap.png')}")


if __name__ == "__main__":
    main()
