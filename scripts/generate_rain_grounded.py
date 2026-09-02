"""
Piece 4 (rain), scene-grounded arm -- our method. Composites all five rain
components in order: wet surface darkening -> wet surface reflections ->
global atmosphere shift -> scene-grounded veiling -> streaks.

See scripts/rain_utils.py for the full per-component math, the calibrated-
constant/seeding design decisions, and attribution (Tremblay 2020 / Weber
2015 for the veiling functional form, Garg & Nayar 2006 for the general
streak-physics concept -- attribution only, nothing adapted from either
codebase; see the rain-rendering/DAF-Net audit).

Usage:
  python3 scripts/generate_rain_grounded.py \
      --image aachen_000000_000019 --split train --city aachen --rain-rate 50.0
"""

import argparse

import numpy as np

from fog_utils import (
    PROJECT_ROOT,
    ATMOSPHERIC_LIGHT,
    disparity_to_pseudo_depth,
    load_clean,
    load_disparity,
    load_seg_labels,
    save_aux,
    save_image,
)
from generate_grounded import DEFAULT_SIGMA
from rain_utils import (
    MAX_STREAK_COUNT,
    RAIN_VEILING_C,
    STREAKS_PER_MM,
    apply_atmosphere_shift,
    apply_reflections,
    apply_wet_saturation,
    apply_wet_shine,
    apply_streaks,
    apply_veiling,
    apply_wet_darkening,
    build_reflection,
    build_reflection_mask,
    build_streak_layer,
    build_wet_mask,
    compute_contact_rows,
    compute_veiling_beta_map,
    derive_streak_seed,
    rain_aux_channels,
    sample_atmosphere_shift,
)

OUT_ROOT = PROJECT_ROOT / "data" / "generated" / "rain_grounded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="aachen_000000_000019", help="image stem, no suffix")
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", default="aachen")
    ap.add_argument("--rain-rate", type=float, default=50.0, help="mm/hr")
    ap.add_argument("--streak-seed", type=int, default=None, help="override the derived streak seed (default: derive from --image)")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="veiling beta-map gaussian smoothing (same default as fog)")
    ap.add_argument("--veiling-c", type=float, default=RAIN_VEILING_C)
    args = ap.parse_args()

    clean = load_clean(args.split, args.city, args.image)
    disparity = load_disparity(args.split, args.city, args.image)
    depth = disparity_to_pseudo_depth(disparity)
    seg = load_seg_labels(args.split, args.city, args.image)

    if seg.shape != depth.shape:
        raise SystemExit(
            f"Shape mismatch: seg {seg.shape} vs depth {depth.shape} -- "
            f"clean/seg/depth must all be the same resolution."
        )

    streak_seed = args.streak_seed if args.streak_seed is not None else derive_streak_seed(args.image)

    # component 1: wet surface darkening
    wet_mask = build_wet_mask(seg, args.rain_rate)
    j_wet = apply_wet_darkening(clean, wet_mask)

    # component 2's mirror is computed here (moved earlier than its
    # original position) because shine (v10) now sources its brightness
    # from `reflection` -- contact_rows/reflection/reflection_mask don't
    # depend on darkening/shine/saturation, only on clean+seg, so this
    # reordering is free.
    contact_rows = compute_contact_rows(seg)
    reflection = build_reflection(clean, contact_rows)
    reflection_mask = build_reflection_mask(seg, args.rain_rate)

    j_shine = apply_wet_shine(j_wet, wet_mask, args.rain_rate, reflection, ATMOSPHERIC_LIGHT)
    j_saturated = apply_wet_saturation(j_shine, wet_mask, args.rain_rate)

    # component 2: wet surface reflections composite onto the
    # darkened+shined+saturated result
    j_wet_reflect = apply_reflections(j_saturated, reflection, reflection_mask)

    # component 3: global atmosphere shift (applied to J_wet_reflect and to A together)
    gamma, blue_boost = sample_atmosphere_shift(args.image, args.rain_rate)
    j_shifted, a_shifted = apply_atmosphere_shift(j_wet_reflect, ATMOSPHERIC_LIGHT, gamma, blue_boost)

    # component 4: scene-grounded veiling (reuses fog's build_beta_map)
    beta_map = compute_veiling_beta_map(seg, depth, args.rain_rate, c=args.veiling_c, sigma=args.sigma)
    i_veiled = apply_veiling(j_shifted, depth, beta_map, a_shifted)

    # component 5: streaks (composited last)
    streak_alpha, streak_brightness = build_streak_layer(depth.shape, depth, args.rain_rate, streak_seed)
    rainy = apply_streaks(i_veiled, streak_alpha, streak_brightness)

    stem = f"{args.image}_rain{args.rain_rate:.1f}_grounded"
    out_dir = OUT_ROOT / args.split / args.city
    save_image(rainy, out_dir / f"{stem}.png")

    # debug/thesis-figure viz of the combined surface-effect mask -- same
    # spirit as fog's _betamap.png, not part of the manifest's formal
    # file_paths schema.
    surface_vis = np.repeat(np.maximum(wet_mask, reflection_mask)[:, :, None], 3, axis=2)
    save_image(surface_vis, out_dir / f"{stem}_surfacemask.png")

    veiling_density, surface_effect = rain_aux_channels(beta_map, depth, wet_mask, reflection_mask)
    save_aux(veiling_density, out_dir / f"{stem}_aux.npz", surface_effect=surface_effect)

    n_streaks = min(int(args.rain_rate * STREAKS_PER_MM), MAX_STREAK_COUNT)
    print(f"rain_rate = {args.rain_rate}, streak_seed = {streak_seed}")
    print(f"gamma = {gamma:.4f}, blue_boost = {blue_boost:.4f}")
    print(f"veiling beta_map range: [{beta_map.min():.3f}, {beta_map.max():.3f}]")
    print(f"n_streaks = {n_streaks}")
    print(f"Saved: {out_dir / (stem + '.png')}")
    print(f"Saved surface mask viz: {out_dir / (stem + '_surfacemask.png')}")
    print(f"Saved aux: {out_dir / (stem + '_aux.npz')}")


if __name__ == "__main__":
    main()
