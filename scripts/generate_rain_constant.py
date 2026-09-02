"""
Piece 4 (rain), baseline arm -- constant-rate rain generation. Mirrors
generate_constant.py's role for fog: no scene-grounding.

BUG FIX (found via a constant-vs-grounded brightness diagnostic, see
commit history): only components 1 (wet darkening) and 2 (reflections)
are grounded-only -- component 3 (atmosphere shift) is a GLOBAL effect
(overcast lighting, blue tint) that applies regardless of whether wet
surfaces are modeled, and previously skipping it here made the constant
arm systematically miss a real brightness/tone change that grounded got,
confounding the arms' comparison with something that had nothing to do
with the wet-surface novel contribution. Constant now applies atmosphere
shift with the SAME gamma/blue_boost as grounded would for this image
(sample_atmosphere_shift depends only on image+rain_rate, not on arm, so
this is automatic -- no extra parameter needed).

Pipeline: clean -> atmosphere shift (component 3) -> veiling (component
4, uniform/non-seg-modified beta) -> streaks (component 5). Components
1-2 skipped.

See scripts/rain_utils.py for the underlying math/citations.

Usage:
  python3 scripts/generate_rain_constant.py \
      --image aachen_000000_000019 --split train --city aachen --rain-rate 50.0
"""

import argparse

from fog_utils import (
    PROJECT_ROOT,
    ATMOSPHERIC_LIGHT,
    disparity_to_pseudo_depth,
    load_clean,
    load_disparity,
    save_aux,
    save_image,
)
from rain_utils import (
    MAX_STREAK_COUNT,
    RAIN_VEILING_C,
    STREAKS_PER_MM,
    apply_atmosphere_shift,
    apply_streaks,
    apply_veiling,
    build_streak_layer,
    compute_veiling_beta_uniform,
    derive_streak_seed,
    rain_aux_channels,
    sample_atmosphere_shift,
)

OUT_ROOT = PROJECT_ROOT / "data" / "generated" / "rain_constant"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="aachen_000000_000019", help="image stem, no suffix")
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", default="aachen")
    ap.add_argument("--rain-rate", type=float, default=50.0, help="mm/hr")
    ap.add_argument("--streak-seed", type=int, default=None, help="override the derived streak seed (default: derive from --image)")
    ap.add_argument("--veiling-c", type=float, default=RAIN_VEILING_C)
    args = ap.parse_args()

    clean = load_clean(args.split, args.city, args.image)
    disparity = load_disparity(args.split, args.city, args.image)
    depth = disparity_to_pseudo_depth(disparity)

    streak_seed = args.streak_seed if args.streak_seed is not None else derive_streak_seed(args.image)

    # component 3: global atmosphere shift -- same gamma/blue_boost the
    # grounded arm would use for this image (depends only on image+rain_rate)
    gamma, blue_boost = sample_atmosphere_shift(args.image, args.rain_rate)
    j_shifted, a_shifted = apply_atmosphere_shift(clean, ATMOSPHERIC_LIGHT, gamma, blue_boost)

    # component 4: uniform (non-seg-modified) veiling
    beta_map = compute_veiling_beta_uniform(args.rain_rate, depth.shape, c=args.veiling_c)
    i_veiled = apply_veiling(j_shifted, depth, beta_map, a_shifted)

    # component 5: streaks
    streak_alpha, streak_brightness = build_streak_layer(depth.shape, depth, args.rain_rate, streak_seed)
    rainy = apply_streaks(i_veiled, streak_alpha, streak_brightness)

    stem = f"{args.image}_rain{args.rain_rate:.1f}_constant"
    out_path = OUT_ROOT / args.split / args.city / f"{stem}.png"
    save_image(rainy, out_path)

    # Aux artifact: channel 0 = veiling density (uniform beta, still varies
    # per-pixel via depth, same as fog's constant arm). Channel 1 = 0
    # (components 1-2 skipped -- legitimately all-zero here, same
    # convention as fog's constant arm).
    veiling_density, surface_effect = rain_aux_channels(beta_map, depth)
    aux_path = out_path.with_name(out_path.stem + "_aux.npz")
    save_aux(veiling_density, aux_path, surface_effect=surface_effect)

    n_streaks = min(int(args.rain_rate * STREAKS_PER_MM), MAX_STREAK_COUNT)
    print(f"rain_rate = {args.rain_rate}, streak_seed = {streak_seed}")
    print(f"gamma = {gamma:.4f}, blue_boost = {blue_boost:.4f}")
    print(f"veiling beta (uniform) = {float(beta_map[0, 0]):.4f}")
    print(f"n_streaks = {n_streaks}")
    print(f"Saved: {out_path}")
    print(f"Saved aux: {aux_path}")


if __name__ == "__main__":
    main()
