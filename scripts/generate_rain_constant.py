"""
Piece 4 (rain), baseline arm -- constant-rate rain generation. Mirrors
generate_constant.py's role for fog: no scene-grounding at all.

Per the locked spec (revised during design review from the original
proposal, which had constant keep the atmosphere-shift step): constant
skips components 1 (wet darkening), 2 (reflections), AND 3 (atmosphere
shift) entirely -- the clean image goes straight into veiling (component
4, uniform/non-seg-modified beta) and then streaks (component 5). Since
there's no atmosphere-shift step here, the fixed, unshifted
ATMOSPHERIC_LIGHT is used as-is for A (no gamma/blue_boost applied).

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
    apply_streaks,
    apply_veiling,
    build_streak_layer,
    compute_veiling_beta_uniform,
    derive_streak_seed,
    rain_aux_channels,
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

    # component 4: uniform (non-seg-modified) veiling -- clean image straight
    # in, fixed unshifted ATMOSPHERIC_LIGHT (components 1-3 skipped)
    beta_map = compute_veiling_beta_uniform(args.rain_rate, depth.shape, c=args.veiling_c)
    i_veiled = apply_veiling(clean, depth, beta_map, ATMOSPHERIC_LIGHT)

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
    print(f"veiling beta (uniform) = {float(beta_map[0, 0]):.4f}")
    print(f"n_streaks = {n_streaks}")
    print(f"Saved: {out_path}")
    print(f"Saved aux: {aux_path}")


if __name__ == "__main__":
    main()
