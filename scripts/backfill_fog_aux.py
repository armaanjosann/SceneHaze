"""
Session A of the fog retrofit: backfill the 3-channel aux .npz for the two
existing fog arms (constant, grounded) across all 3,475 images already on
disk. One-time batch operation, not part of the normal generate_*.py flow
(those now save aux for any NEW image going forward -- this script exists
only to backfill the ones generated before that capability existed).

Does NOT touch RGB or the existing _betamap.png -- those are untouched,
verified by a separate bit-identical pre-flight check (recompute one image
fully via apply_asm and diff against on-disk RGB) run before this script,
not repeated here. Since that check passed, this script trusts
build_beta_map's determinism (pure function of seg/depth/beta_base/sigma,
no randomness) and skips redoing the full ASM math (no `clean` image is
even loaded -- veiling density depends only on beta and depth, not pixel
content) for the other 3,474 images, purely for speed.

Iterates the manifest in its existing canonical order (split -> city ->
image, preserved unchanged through the v1->v2 migration), reusing the
manifest's own file_paths.fog.<arm>.aux strings as the write targets
rather than reconstructing them independently.

Resumable: skips an image if both its aux files already exist.

Usage:
  python3 scripts/backfill_fog_aux.py
"""

import shutil
import time

import numpy as np

from fog_utils import PROJECT_ROOT, disparity_to_pseudo_depth, load_disparity, load_seg_labels, save_aux
from generate_grounded import build_beta_map, DEFAULT_SIGMA
from manifest_io import read_manifest, DEFAULT_MANIFEST_PATH

MIN_FREE_GB = 5.0


def main():
    manifest = read_manifest(DEFAULT_MANIFEST_PATH)
    print(f"Loaded manifest: {len(manifest)} images")

    n_processed = 0
    n_skipped = 0
    t_start = time.time()

    for i, entry in enumerate(manifest, 1):
        split, city, image = entry["split"], entry["city"], entry["image"]
        beta_base = entry["fog"]["beta_base"]

        constant_aux_path = PROJECT_ROOT / entry["file_paths"]["fog"]["constant"]["aux"]
        grounded_aux_path = PROJECT_ROOT / entry["file_paths"]["fog"]["grounded"]["aux"]

        if constant_aux_path.exists() and grounded_aux_path.exists():
            n_skipped += 1
            continue

        free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"\nABORTING: only {free_gb:.2f}GB free (< {MIN_FREE_GB}GB).")
            print(f"Stopped before [{i}/{len(manifest)}] {city}/{image}. Rerun to resume.")
            break

        if i % 250 == 0 or i == 1:
            print(f"[{i}/{len(manifest)}] {city}/{image}")

        disparity = load_disparity(split, city, image)
        depth = disparity_to_pseudo_depth(disparity)
        seg = load_seg_labels(split, city, image)

        if seg.shape != depth.shape:
            print(f"  WARNING: shape mismatch (seg={seg.shape}, depth={depth.shape}) -- skipping {city}/{image}")
            continue

        beta_map = build_beta_map(seg, depth, beta_base, DEFAULT_SIGMA)
        veiling_grounded = 1.0 - np.exp(-beta_map * depth)
        veiling_constant = 1.0 - np.exp(-beta_base * depth)

        save_aux(veiling_constant, constant_aux_path)
        save_aux(veiling_grounded, grounded_aux_path)

        n_processed += 1

    elapsed = time.time() - t_start
    print(f"\nDone. Processed {n_processed}, skipped {n_skipped} (already existed), "
          f"of {len(manifest)} considered.")
    if n_processed:
        print(f"Elapsed: {elapsed:.1f}s ({elapsed/n_processed:.3f}s/image)")


if __name__ == "__main__":
    main()
