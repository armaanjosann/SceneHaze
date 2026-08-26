"""
Piece 2 — scene-grounded ASM fog generation (our method).

Same ASM equation as Piece 1, but beta becomes a per-pixel map instead of a
scalar: beta(x) = beta_base * category_modifier(x), where the modifier comes
from Cityscapes ground-truth semantic labels (_gtFine_labelIds.png).

  Sky              (23)              x0.2              flat — fog thins above the boundary layer
                                                          regardless of this pixel's own depth (sky
                                                          is always effectively "far")
  Ground/road      (7,8,9,10)        x(1.0 + 0.5*depth)  depth-scaled by default (see below)
  Low vegetation   (21,22)           x(1.0 + 0.1*depth)  depth-scaled by default
  Everything else  (incl. structures) x1.0                default — no change from baseline

  (Gaussian smoothing sigma=20 by default. All four values are FID-optimized
  against real NH-HAZE fog — see scripts/fid_calibration.py.)

Depth-scaled modifiers (default): fog accumulates with distance beyond what
the transmission term t(x)=exp(-beta*depth) already captures on its own — a
road pixel right under the hood shouldn't get the same "ground pools fog"
boost as one 500m away. So the category boost itself scales with this
pixel's normalized pseudo-depth: close pixels get only a small fraction of
the boost, far pixels get the full boost. Sky is excluded from this scaling
(it's flat) since "distance" isn't a meaningful concept for sky pixels the
way it is for ground.

Pass --flat-modifiers to revert to the original flat multipliers (x1.3,
x1.1 regardless of depth) — kept as an ablation baseline, not the default.

("Water" has no Cityscapes label — it's added later via SegFormer/ADE20K
when we move beyond Cityscapes.)

The per-pixel beta map is then Gaussian-smoothed (sigma=15 by default) so
category boundaries don't create harsh fog edges — real fog doesn't respect
segmentation masks.

Optional --turbulence layers small-scale organic variation on top of that
large-scale structure: the beta map is multiplied by smooth noise centered
at 1.0 (see fog_utils.apply_turbulence). NOT part of the primary method —
kept as an off-by-default ablation option (does turbulence help or hurt on
top of scene-grounding?), since adding randomness back on top would muddy
the "scene-grounded beats random" comparison against MCBM.

Usage:
  python3 scripts/generate_grounded.py \
      --image aachen_000000_000019 --split train --city aachen --beta-base 1.0

  # ablation: flat category multipliers instead of depth-scaled
  python3 scripts/generate_grounded.py \
      --image aachen_000000_000019 --split train --city aachen --beta-base 1.0 \
      --flat-modifiers

  # ablation: turbulence on top of (depth-scaled, by default) scene-grounding
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

# FID-optimized (see scripts/fid_calibration.py, results/metrics/fid_calibration_results.csv):
# swept sky/road/veg/sigma against real NH-HAZE fog on 50 diverse Cityscapes
# images; combined_best_guess (0.2/1.5/1.1/sigma=20) won with FID=292.8 vs.
# 300.9 for the original hand-picked values (0.3/1.3/1.1/sigma=15).
SKY_MODIFIER = 0.2
GROUND_MODIFIER = 1.5
VEGETATION_MODIFIER = 1.1
DEFAULT_SIGMA = 20.0


def build_beta_map(
    seg: np.ndarray,
    depth: np.ndarray,
    beta_base: float,
    sigma: float,
    flat_modifiers: bool = False,
    sky_mod: float = SKY_MODIFIER,
    ground_mod: float = GROUND_MODIFIER,
    veg_mod: float = VEGETATION_MODIFIER,
) -> np.ndarray:
    beta_map = np.full(seg.shape, beta_base, dtype=np.float32)

    sky_mask = np.isin(seg, list(SKY_IDS))
    ground_mask = np.isin(seg, list(GROUND_IDS))
    veg_mask = np.isin(seg, list(VEGETATION_IDS))

    # Sky stays flat: "close vs. far" isn't a meaningful distinction for sky
    # pixels the way it is for ground, so no depth-scaling here either way.
    beta_map[sky_mask] *= sky_mod

    if flat_modifiers:
        beta_map[ground_mask] *= ground_mod
        beta_map[veg_mask] *= veg_mod
    else:
        # Depth-scaled: close pixels get only a small fraction of the
        # category boost, distant pixels get the full boost. Reuses the
        # same normalized pseudo-depth already fed into the ASM equation.
        beta_map[ground_mask] *= 1.0 + (ground_mod - 1.0) * depth[ground_mask]
        beta_map[veg_mask] *= 1.0 + (veg_mod - 1.0) * depth[veg_mask]

    return gaussian_filter(beta_map, sigma=sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="aachen_000000_000019", help="image stem, no suffix")
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", default="aachen")
    ap.add_argument("--beta-base", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="gaussian smoothing of beta map (FID-optimized default)")
    ap.add_argument("--sky-mod", type=float, default=SKY_MODIFIER, help="sky category modifier (flat, always)")
    ap.add_argument("--road-mod", type=float, default=GROUND_MODIFIER, help="ground/road category modifier (depth-scaled by default)")
    ap.add_argument("--veg-mod", type=float, default=VEGETATION_MODIFIER, help="vegetation category modifier (depth-scaled by default)")
    ap.add_argument("--flat-modifiers", action="store_true", help="ablation: flat category multipliers instead of the default depth-scaled ones")
    ap.add_argument("--turbulence", action="store_true", help="ablation: layer small-scale organic noise on top of the scene-grounded beta map (off by default -- not part of the primary method)")
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

    beta_map = build_beta_map(
        seg, depth, args.beta_base, args.sigma, args.flat_modifiers,
        sky_mod=args.sky_mod, ground_mod=args.road_mod, veg_mod=args.veg_mod,
    )

    stem = f"{args.image}_betabase{args.beta_base:.2f}_grounded"
    if args.flat_modifiers:
        stem += "_flatmod"
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

    print(f"beta_base = {args.beta_base}, sigma = {args.sigma}, flat_modifiers = {args.flat_modifiers}")
    print(f"modifiers: sky={args.sky_mod}, road={args.road_mod}, veg={args.veg_mod}")
    if args.turbulence:
        print(f"turbulence: strength={args.turb_strength}, scale={args.turb_scale}, seed={args.turb_seed}")
    print(f"beta_map range: [{beta_map.min():.3f}, {beta_map.max():.3f}]")
    print(f"Saved: {out_dir / (stem + '.png')}")
    print(f"Saved beta map viz: {out_dir / (stem + '_betamap.png')}")


if __name__ == "__main__":
    main()
