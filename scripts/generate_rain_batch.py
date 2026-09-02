"""
Batch-generates rain (constant + grounded) RGB + aux for all manifest
entries, one image at a time, both arms generated together per image --
mirrors generate_dataset.py's combined-pass pattern (clean/depth/seg
loaded once, reused for both arms) rather than generate_shuffled_batch.py's
single-arm-retrofit pattern, since rain's two arms start from zero at the
same time here -- no reason to load shared inputs twice.

Depends on scripts/populate_rain_params.py having already run (raises
SystemExit if rain.rain_rate / rain.streak_seed / file_paths.rain are
null for an entry -- same defensive pattern as generate_shuffled_batch.py's
shuffle_seed check).

Per-arm resumable: skips an image entirely if BOTH arms' rgb+aux already
exist; generates only the missing arm(s) otherwise. Per-image disk-space
check, progress print every 100, elapsed/image timing + full-dataset
extrapolation at the end -- same shape as every prior batch script in this
project.

Usage:
  python3 scripts/generate_rain_batch.py             # full batch
  python3 scripts/generate_rain_batch.py --limit 10  # first N entries only
"""

import argparse
import shutil
import time

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
from manifest_io import read_manifest, DEFAULT_MANIFEST_PATH
from rain_utils import (
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
    compute_veiling_beta_uniform,
    rain_aux_channels,
    sample_atmosphere_shift,
)

MIN_FREE_GB = 5.0


def render_constant(clean, depth, rain_rate, streak_seed):
    """Components 4 (uniform veiling) + 5 (streaks) only -- see
    generate_rain_constant.py for the single-image CLI equivalent of this
    same sequence."""
    beta_map = compute_veiling_beta_uniform(rain_rate, depth.shape)
    i_veiled = apply_veiling(clean, depth, beta_map, ATMOSPHERIC_LIGHT)
    streak_alpha, streak_brightness = build_streak_layer(depth.shape, depth, rain_rate, streak_seed)
    rgb = apply_streaks(i_veiled, streak_alpha, streak_brightness)
    veiling_density, surface_effect = rain_aux_channels(beta_map, depth)
    return rgb, veiling_density, surface_effect


def render_grounded(clean, depth, seg, image, rain_rate, streak_seed):
    """All five components in order -- see generate_rain_grounded.py for
    the single-image CLI equivalent of this same sequence."""
    wet_mask = build_wet_mask(seg, rain_rate)
    j_wet = apply_wet_darkening(clean, wet_mask)

    # moved earlier than its original position -- shine (v10) sources
    # brightness from `reflection`, which doesn't depend on darkening/
    # shine/saturation, only on clean+seg, so this reordering is free.
    contact_rows = compute_contact_rows(seg)
    reflection = build_reflection(clean, contact_rows)
    reflection_mask = build_reflection_mask(seg, rain_rate)

    j_shine = apply_wet_shine(j_wet, wet_mask, rain_rate, reflection, ATMOSPHERIC_LIGHT)
    j_saturated = apply_wet_saturation(j_shine, wet_mask, rain_rate)
    j_wet_reflect = apply_reflections(j_saturated, reflection, reflection_mask)

    gamma, blue_boost = sample_atmosphere_shift(image, rain_rate)
    j_shifted, a_shifted = apply_atmosphere_shift(j_wet_reflect, ATMOSPHERIC_LIGHT, gamma, blue_boost)

    beta_map = compute_veiling_beta_map(seg, depth, rain_rate, sigma=DEFAULT_SIGMA)
    i_veiled = apply_veiling(j_shifted, depth, beta_map, a_shifted)

    streak_alpha, streak_brightness = build_streak_layer(depth.shape, depth, rain_rate, streak_seed)
    rgb = apply_streaks(i_veiled, streak_alpha, streak_brightness)

    veiling_density, surface_effect = rain_aux_channels(beta_map, depth, wet_mask, reflection_mask)
    return rgb, veiling_density, surface_effect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N manifest entries")
    args = ap.parse_args()

    manifest = read_manifest(DEFAULT_MANIFEST_PATH)
    full_total = len(manifest)
    entries = manifest[: args.limit] if args.limit else manifest

    print(f"Loaded manifest: {full_total} entries; processing {len(entries)}"
          + (f" (--limit {args.limit})" if args.limit else ""))

    n_processed = 0
    n_skipped = 0
    t_start = time.time()

    for i, entry in enumerate(entries, 1):
        split, city, image = entry["split"], entry["city"], entry["image"]
        rain_rate = entry["rain"]["rain_rate"]
        streak_seed = entry["rain"]["streak_seed"]
        rain_paths = entry["file_paths"]["rain"]

        if rain_rate is None or streak_seed is None or rain_paths is None:
            raise SystemExit(
                f"{city}/{image}: rain.rain_rate / rain.streak_seed / file_paths.rain is null. "
                f"Run scripts/populate_rain_params.py first."
            )

        constant_rgb = PROJECT_ROOT / rain_paths["constant"]["rgb"]
        constant_aux = PROJECT_ROOT / rain_paths["constant"]["aux"]
        grounded_rgb = PROJECT_ROOT / rain_paths["grounded"]["rgb"]
        grounded_aux = PROJECT_ROOT / rain_paths["grounded"]["aux"]

        need_constant = not (constant_rgb.exists() and constant_aux.exists())
        need_grounded = not (grounded_rgb.exists() and grounded_aux.exists())

        if not need_constant and not need_grounded:
            n_skipped += 1
            continue

        free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"\nABORTING: only {free_gb:.2f}GB free (< {MIN_FREE_GB}GB).")
            print(f"Stopped before [{i}/{len(entries)}] {city}/{image}. Rerun to resume.")
            break

        arms_needed = [a for a, need in (("constant", need_constant), ("grounded", need_grounded)) if need]
        if i % 100 == 0 or i == 1:
            print(f"[{i}/{len(entries)}] {city}/{image}  (rain_rate={rain_rate:.1f}, arms: {arms_needed})")

        clean = load_clean(split, city, image)
        disparity = load_disparity(split, city, image)
        depth = disparity_to_pseudo_depth(disparity)
        seg = load_seg_labels(split, city, image)

        if seg.shape != depth.shape:
            print(f"  WARNING: shape mismatch (seg={seg.shape}, depth={depth.shape}) -- skipping {city}/{image}")
            continue

        if need_constant:
            rgb, veiling_density, surface_effect = render_constant(clean, depth, rain_rate, streak_seed)
            save_image(rgb, constant_rgb)
            save_aux(veiling_density, constant_aux, surface_effect=surface_effect)

        if need_grounded:
            rgb, veiling_density, surface_effect = render_grounded(clean, depth, seg, image, rain_rate, streak_seed)
            save_image(rgb, grounded_rgb)
            save_aux(veiling_density, grounded_aux, surface_effect=surface_effect)

        n_processed += 1

    elapsed = time.time() - t_start
    print(f"\nDone this run. Processed {n_processed}, skipped {n_skipped} (both arms already existed), "
          f"of {len(entries)} considered.")
    if n_processed:
        per_image = elapsed / n_processed
        print(f"Elapsed: {elapsed:.1f}s ({per_image:.3f}s/image)")
        if args.limit:
            est_full = per_image * full_total
            print(f"Extrapolated estimate for all {full_total} images: "
                  f"~{est_full/60:.1f} min ({est_full/3600:.2f} hours)")


if __name__ == "__main__":
    main()
