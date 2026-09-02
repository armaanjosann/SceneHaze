"""
Session B of the fog retrofit: compute per-arm stats (mean_abs_delta,
mean_transmission, grounding_depth_corr, grounding_seg_mi) for the
constant and grounded fog arms, and write them into the manifest.

Generates nothing -- reads existing RGB, aux .npz (Session A), depth, and
segmentation, computes four numbers per arm per image, and populates
data/dataset_manifest.json's fog.<arm>.stats fields via manifest_io's
lock/atomic-write pattern.

Shuffled arm is skipped entirely -- doesn't exist yet.

Resumable: an entry is skipped if fog.constant.stats.mean_abs_delta is
already non-null.

Locking note: the manifest is read once at the START (outside the lock),
since the compute phase over 3,475 images takes long enough that holding
the file lock for the whole duration would be poor lock hygiene (blocks
any other script from touching the manifest for that whole time, for no
real benefit in a single-agent context). The lock is only taken at the
END, around a final re-read-and-merge-then-write: re-read the manifest
inside the lock, merge this run's freshly-computed stats into that
CURRENT on-disk state (not the possibly-stale copy read at the start),
then atomic_write_manifest(). This is deliberately stricter than "read
once, write the in-memory copy back" -- it avoids a lost-update race if
anything else touched the manifest while this script was computing,
without paying for a multi-minute lock hold.

Usage:
  python3 scripts/compute_fog_stats.py            # full batch
  python3 scripts/compute_fog_stats.py --limit 5   # first 5 entries only, verbose
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score

from fog_utils import PROJECT_ROOT, load_clean, load_disparity, load_seg_labels, disparity_to_pseudo_depth
from generate_grounded import build_beta_map, DEFAULT_SIGMA
from manifest_io import read_manifest, atomic_write_manifest, acquire_lock, DEFAULT_MANIFEST_PATH

N_MI_BINS = 50
ZERO_VARIANCE_EPS = 1e-8


def load_rgb01(path: Path) -> np.ndarray:
    """Inverse of fog_utils.save_image: uint8 PNG -> float32 in [0,1]."""
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def bin_continuous(arr: np.ndarray, n_bins: int = N_MI_BINS) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < ZERO_VARIANCE_EPS:
        return np.zeros(arr.shape, dtype=np.int64)
    bins = np.floor((arr - lo) / (hi - lo) * n_bins).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)


def compute_grounding_depth_corr(beta_map: np.ndarray, depth: np.ndarray) -> float:
    if beta_map.max() - beta_map.min() < ZERO_VARIANCE_EPS:
        return 0.0
    return float(pearsonr(beta_map.ravel(), depth.ravel())[0])


def compute_grounding_seg_mi(beta_map: np.ndarray, seg: np.ndarray) -> float:
    if beta_map.max() - beta_map.min() < ZERO_VARIANCE_EPS:
        # single unique value (constant arm) -> MI is exactly 0 by
        # construction; skip the sklearn call entirely rather than binning
        # a degenerate constant array.
        return 0.0
    binned = bin_continuous(beta_map)
    mi_nats = mutual_info_score(seg.ravel(), binned.ravel())
    return float(mi_nats / math.log(2))  # nats -> bits


def compute_arm_stats(foggy_rgb: np.ndarray, clean_rgb: np.ndarray, beta_map: np.ndarray,
                       depth: np.ndarray, seg: np.ndarray, aux_path: Path) -> dict:
    mean_abs_delta = float(np.mean(np.abs(foggy_rgb - clean_rgb)))

    aux = np.load(aux_path)["aux"]
    mean_transmission = float(np.mean(1.0 - aux[:, :, 0].astype(np.float32)))

    return {
        "mean_abs_delta": mean_abs_delta,
        "grounding_depth_corr": compute_grounding_depth_corr(beta_map, depth),
        "grounding_seg_mi": compute_grounding_seg_mi(beta_map, seg),
        "mean_transmission": mean_transmission,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N manifest entries (for sample runs)")
    args = ap.parse_args()

    manifest = read_manifest(DEFAULT_MANIFEST_PATH)
    total = len(manifest)
    entries = manifest[: args.limit] if args.limit else manifest

    print(f"Loaded manifest: {total} entries; processing {len(entries)}"
          + (f" (--limit {args.limit})" if args.limit else ""))

    n_processed = 0
    n_skipped = 0

    for i, entry in enumerate(entries, 1):
        split, city, image = entry["split"], entry["city"], entry["image"]

        if entry["fog"]["constant"]["stats"]["mean_abs_delta"] is not None:
            n_skipped += 1
            continue

        if i % 100 == 0 or i == 1:
            print(f"[{i}/{len(entries)}] {city}/{image}")

        beta_base = entry["fog"]["beta_base"]

        clean = load_clean(split, city, image)
        disparity = load_disparity(split, city, image)
        depth = disparity_to_pseudo_depth(disparity)
        seg = load_seg_labels(split, city, image)

        if seg.shape != depth.shape:
            print(f"  WARNING: shape mismatch (seg={seg.shape}, depth={depth.shape}) -- skipping {city}/{image}")
            continue

        # --- constant arm ---
        constant_rgb_path = PROJECT_ROOT / entry["file_paths"]["fog"]["constant"]["rgb"]
        constant_aux_path = PROJECT_ROOT / entry["file_paths"]["fog"]["constant"]["aux"]
        constant_foggy = load_rgb01(constant_rgb_path)
        beta_map_constant = np.full(depth.shape, beta_base, dtype=np.float32)
        constant_stats = compute_arm_stats(constant_foggy, clean, beta_map_constant, depth, seg, constant_aux_path)

        # --- grounded arm ---
        grounded_rgb_path = PROJECT_ROOT / entry["file_paths"]["fog"]["grounded"]["rgb"]
        grounded_aux_path = PROJECT_ROOT / entry["file_paths"]["fog"]["grounded"]["aux"]
        grounded_foggy = load_rgb01(grounded_rgb_path)
        beta_map_grounded = build_beta_map(seg, depth, beta_base, DEFAULT_SIGMA)
        grounded_stats = compute_arm_stats(grounded_foggy, clean, beta_map_grounded, depth, seg, grounded_aux_path)

        entry["fog"]["constant"]["stats"] = constant_stats
        entry["fog"]["grounded"]["stats"] = grounded_stats

        if args.limit:
            print(f"  constant: {constant_stats}")
            print(f"  grounded: {grounded_stats}")

        n_processed += 1

    print(f"\nProcessed {n_processed}, skipped {n_skipped} (already had stats), of {len(entries)} considered.")

    if n_processed == 0:
        print("Nothing new to write -- manifest unchanged.")
        return

    with acquire_lock(DEFAULT_MANIFEST_PATH) as token:
        current = read_manifest(DEFAULT_MANIFEST_PATH)
        current_by_key = {(e["split"], e["city"], e["image"]): e for e in current}
        n_merged = 0
        for entry in entries:
            key = (entry["split"], entry["city"], entry["image"])
            if key in current_by_key and entry["fog"]["constant"]["stats"]["mean_abs_delta"] is not None:
                current_by_key[key]["fog"]["constant"]["stats"] = entry["fog"]["constant"]["stats"]
                current_by_key[key]["fog"]["grounded"]["stats"] = entry["fog"]["grounded"]["stats"]
                n_merged += 1
        atomic_write_manifest(current, DEFAULT_MANIFEST_PATH, lock_token=token)

    print(f"Manifest updated: {DEFAULT_MANIFEST_PATH} ({n_merged} entries merged in)")


if __name__ == "__main__":
    main()
