"""
Full-dataset batch generation: constant-beta AND scene-grounded fog for all
usable Cityscapes images (train + val = 3,475 -- test excluded, its
gtFine labels are dummy placeholders, see project context).

MCBM is NOT generated here -- that happens later on Kaya HPC with
HazeFlow's own code.

Single source of truth is data/dataset_manifest.json: one entry per image
with a seeded beta_base ~ Uniform(0.4, 2.0) (seed=42 by default), shared
identically between constant-beta and scene-grounded so the only thing
that differs between the two methods is the beta computation itself, not
the severity.

Per image, in one pass (depth loaded/computed once, reused for both
methods):
  1. load clean image
  2. ensure depth (compute + cache as float32 .npy if not already cached)
  3. load segmentation labels
  4. generate + save constant-beta fog
  5. generate + save scene-grounded fog (locked-in FID-optimized defaults:
     sky=0.2, road=1.5, veg=1.1, sigma=20, depth-scaled modifiers) + its
     beta map visualization

Resumable: if both the constant and grounded PNG already exist for an
image, it's skipped without touching depth/seg/clean at all. Safe to
Ctrl-C and rerun.

Usage:
  # dry run on the first 10 images only
  python3 scripts/generate_dataset.py --limit 10

  # full run (long -- run under caffeinate so the Mac doesn't sleep):
  caffeinate -i python3 scripts/generate_dataset.py
"""

import argparse
import json
import shutil
import time

import numpy as np

from extract_depth import get_device, get_pipeline, ensure_depth
from fog_utils import (
    PROJECT_ROOT,
    CLEAN_ROOT,
    apply_asm,
    disparity_to_pseudo_depth,
    load_clean,
    load_seg_labels,
    save_image,
)
from generate_grounded import build_beta_map, DEFAULT_SIGMA

MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"
CONSTANT_ROOT = PROJECT_ROOT / "data" / "generated" / "constant_beta"
GROUNDED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded"

SPLITS = ["train", "val"]  # test excluded: dummy/placeholder segmentation labels only


def build_manifest(seed: int, beta_range: tuple) -> list:
    entries = []
    for split in SPLITS:
        split_root = CLEAN_ROOT / split
        cities = sorted(p.name for p in split_root.iterdir() if p.is_dir())
        for city in cities:
            images = sorted((split_root / city).glob("*_leftImg8bit.png"))
            for img_path in images:
                stem = img_path.stem.replace("_leftImg8bit", "")
                entries.append({"split": split, "city": city, "image": stem})

    # Deterministic order (split, then city, then image, all alphabetical)
    # fixed above by construction -- so a fixed seed always assigns the same
    # beta_base to the same image, regardless of how many times this is run.
    rng = np.random.default_rng(seed)
    betas = rng.uniform(beta_range[0], beta_range[1], size=len(entries))
    for entry, beta in zip(entries, betas):
        entry["beta_base"] = round(float(beta), 4)
    return entries


def load_or_build_manifest(seed: int, beta_range: tuple) -> list:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as fh:
            manifest = json.load(fh)
        print(f"Loaded existing manifest: {len(manifest)} images ({MANIFEST_PATH})")
        return manifest

    print(f"Building new manifest (seed={seed}, beta_range={beta_range})...")
    manifest = build_manifest(seed, beta_range)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Saved manifest: {len(manifest)} images -> {MANIFEST_PATH}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N manifest images (for dry runs)")
    ap.add_argument("--seed", type=int, default=42, help="seed for beta_base assignment (only used if manifest doesn't exist yet)")
    ap.add_argument("--beta-range", type=float, nargs=2, default=[0.4, 2.0])
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="abort cleanly if free disk space drops below this")
    args = ap.parse_args()

    manifest = load_or_build_manifest(args.seed, tuple(args.beta_range))
    full_total = len(manifest)
    if args.limit:
        manifest = manifest[: args.limit]

    print(f"Processing {len(manifest)} of {full_total} manifest images"
          + (f" (--limit {args.limit})" if args.limit else ""))

    device = get_device()
    print(f"Depth device: {device}")
    depth_pipe = get_pipeline(device)

    n_processed = 0
    n_skipped = 0
    t_start = time.time()

    for i, entry in enumerate(manifest, 1):
        split, city, image, beta_base = entry["split"], entry["city"], entry["image"], entry["beta_base"]

        constant_path = CONSTANT_ROOT / split / city / f"{image}_beta{beta_base:.2f}_constant.png"
        grounded_stem = f"{image}_betabase{beta_base:.2f}_grounded"
        grounded_path = GROUNDED_ROOT / split / city / f"{grounded_stem}.png"
        betamap_path = GROUNDED_ROOT / split / city / f"{grounded_stem}_betamap.png"

        if constant_path.exists() and grounded_path.exists():
            n_skipped += 1
            continue

        free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
        if free_gb < args.min_free_gb:
            print(f"\nABORTING: only {free_gb:.2f}GB free (< --min-free-gb {args.min_free_gb}GB).")
            print(f"Stopped before [{i}/{len(manifest)}] {city}/{image}. Already-written images are untouched.")
            print("Free up space and rerun the same command -- it will resume from here.")
            break

        print(f"[{i}/{len(manifest)}] processing {city}/{image} (β={beta_base:.2f})")

        clean = load_clean(split, city, image)
        disparity = ensure_depth(depth_pipe, split, city, image)
        depth = disparity_to_pseudo_depth(disparity)
        seg = load_seg_labels(split, city, image)

        if seg.shape != depth.shape:
            print(f"  WARNING: shape mismatch (seg={seg.shape}, depth={depth.shape}) -- skipping this image")
            continue

        constant_foggy = apply_asm(clean, depth, beta_base)
        save_image(constant_foggy, constant_path)

        beta_map = build_beta_map(seg, depth, beta_base, DEFAULT_SIGMA)  # locked-in FID-optimized defaults
        grounded_foggy = apply_asm(clean, depth, beta_map)
        save_image(grounded_foggy, grounded_path)

        beta_vis = (beta_map - beta_map.min()) / (beta_map.max() - beta_map.min() + 1e-8)
        save_image(np.repeat(beta_vis[:, :, None], 3, axis=2), betamap_path)

        n_processed += 1

    elapsed = time.time() - t_start
    print(f"\nDone this run. Processed {n_processed}, skipped {n_skipped} (already existed), "
          f"of {len(manifest)} considered.")
    if n_processed:
        per_image = elapsed / n_processed
        print(f"Elapsed: {elapsed:.1f}s ({per_image:.2f}s/image)")
        if args.limit:
            est_full = per_image * full_total
            print(f"Extrapolated estimate for all {full_total} images: "
                  f"~{est_full/60:.1f} min ({est_full/3600:.2f} hours)")


if __name__ == "__main__":
    main()
