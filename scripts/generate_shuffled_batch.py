"""
Session C of the fog retrofit, phase 2: batch-generate the shuffled arm
(RGB + aux) for manifest entries, mirroring generate_shuffled.py's logic
but iterating the whole manifest like backfill_fog_aux.py/generate_dataset.py
did for the other arms.

Unlike backfill_fog_aux.py (which only added aux to already-existing RGB),
shuffled RGB doesn't exist at all yet -- this script generates both RGB and
aux together per image, from cached depth/segmentation (no depth-model
inference needed) plus the manifest's shuffle_seed (populated separately by
populate_shuffle_seeds.py -- must be run first).

Iterates the manifest in canonical order, reuses file_paths.fog.shuffled.
{rgb,aux} as write targets. Resumable: skips an image if both files already
exist.

Usage:
  python3 scripts/generate_shuffled_batch.py             # full batch
  python3 scripts/generate_shuffled_batch.py --limit 10  # first N entries only
"""

import argparse
import random
import shutil
import time

import numpy as np

from fog_utils import PROJECT_ROOT, apply_asm, disparity_to_pseudo_depth, load_clean, load_disparity, load_seg_labels, save_aux, save_image
from generate_grounded import build_beta_map, DEFAULT_SIGMA
from generate_shuffled import compute_derangements, GLOBAL_SEED
from generate_grounded import SKY_MODIFIER, GROUND_MODIFIER, VEGETATION_MODIFIER
from manifest_io import read_manifest, DEFAULT_MANIFEST_PATH

MIN_FREE_GB = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N manifest entries")
    args = ap.parse_args()

    manifest = read_manifest(DEFAULT_MANIFEST_PATH)
    full_total = len(manifest)
    entries = manifest[: args.limit] if args.limit else manifest

    print(f"Loaded manifest: {full_total} entries; processing {len(entries)}"
          + (f" (--limit {args.limit})" if args.limit else ""))

    derangements = compute_derangements(SKY_MODIFIER, GROUND_MODIFIER, VEGETATION_MODIFIER)

    n_processed = 0
    n_skipped = 0
    t_start = time.time()

    for i, entry in enumerate(entries, 1):
        split, city, image = entry["split"], entry["city"], entry["image"]
        beta_base = entry["fog"]["beta_base"]
        shuffle_seed = entry["fog"]["shuffle_seed"]

        if shuffle_seed is None:
            raise SystemExit(
                f"{city}/{image}: fog.shuffle_seed is null. "
                f"Run scripts/populate_shuffle_seeds.py first."
            )

        rgb_path = PROJECT_ROOT / entry["file_paths"]["fog"]["shuffled"]["rgb"]
        aux_path = PROJECT_ROOT / entry["file_paths"]["fog"]["shuffled"]["aux"]

        if rgb_path.exists() and aux_path.exists():
            n_skipped += 1
            continue

        free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"\nABORTING: only {free_gb:.2f}GB free (< {MIN_FREE_GB}GB).")
            print(f"Stopped before [{i}/{len(entries)}] {city}/{image}. Rerun to resume.")
            break

        if i % 100 == 0 or i == 1:
            print(f"[{i}/{len(entries)}] {city}/{image}")

        clean = load_clean(split, city, image)
        disparity = load_disparity(split, city, image)
        depth = disparity_to_pseudo_depth(disparity)
        seg = load_seg_labels(split, city, image)

        if seg.shape != depth.shape:
            print(f"  WARNING: shape mismatch (seg={seg.shape}, depth={depth.shape}) -- skipping {city}/{image}")
            continue

        rng = random.Random(shuffle_seed)
        permutation = rng.choice(derangements)

        beta_map = build_beta_map(
            seg, depth, beta_base, DEFAULT_SIGMA, flat_modifiers=False,
            sky_mod=permutation["sky"], ground_mod=permutation["ground"], veg_mod=permutation["veg"],
        )
        foggy = apply_asm(clean, depth, beta_map)
        save_image(foggy, rgb_path)

        veiling_density = 1.0 - np.exp(-beta_map * depth)
        save_aux(veiling_density, aux_path)

        n_processed += 1

    elapsed = time.time() - t_start
    print(f"\nDone. Processed {n_processed}, skipped {n_skipped} (already existed), "
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
