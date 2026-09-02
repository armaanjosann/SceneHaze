"""
Sessions B+C of the fog retrofit: compute per-arm stats (mean_abs_delta,
mean_transmission, grounding_depth_corr, grounding_ref_corr) for the
constant, grounded, AND shuffled fog arms, and write them into the
manifest.

Generates nothing -- reads existing RGB, aux .npz, depth, and
segmentation, computes four numbers per arm per image, and populates
data/dataset_manifest.json's fog.<arm>.stats fields via manifest_io's
lock/atomic-write pattern.

grounding_ref_corr replaces an earlier grounding_seg_mi (mutual
information vs. raw segmentation labels) that turned out to be
structurally incapable of distinguishing grounded from shuffled: MI is
invariant to any bijective relabeling of either variable, and shuffling
IS exactly a bijective relabeling of which modifier value lands on which
category -- the underlying category PARTITION (which pixels are
sky/ground/veg) is identical in both arms, so MI(seg, beta_map) comes out
statistically identical whether the "correct" or a shuffled mapping was
used. Verified this empirically (not just theoretically) before deciding
to replace it -- see commit history. grounding_ref_corr instead measures
Pearson correlation between THIS arm's beta_map and the CANONICAL
grounded beta_map for the same image (computed once per image, reused as
the reference for all three arms). Correlation is not relabeling-
invariant -- it's sensitive to the actual per-pixel values, not just
whether SOME informative partition exists. Expected: grounded ~1.0 (it's
compared against itself), constant ~0 (no variance), shuffled distinctly
lower (low/negative, varies by which categories got swapped).

Per-arm resumability: for each entry, only the arms whose
stats.mean_abs_delta is still null get (re)computed. An arm is silently
skipped (left null) if its RGB/aux files don't exist yet on disk -- this
makes the script safe to run BEFORE shuffled generation has happened
(computes constant+grounded, leaves shuffled null) and again AFTER
(picks up shuffled), without needing a --arms flag to tell it which
phase it's in.

Locking note: unchanged from Session B -- manifest read once at the
START (outside any lock), lock only taken at the END around a
re-read-and-merge-then-write, to avoid holding the file lock for the
whole multi-minute compute phase.

Usage:
  python3 scripts/compute_fog_stats.py            # full batch
  python3 scripts/compute_fog_stats.py --limit 5   # first 5 entries only, verbose
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import pearsonr

from fog_utils import PROJECT_ROOT, load_clean, load_disparity, load_seg_labels, disparity_to_pseudo_depth
from generate_grounded import build_beta_map, DEFAULT_SIGMA, SKY_MODIFIER, GROUND_MODIFIER, VEGETATION_MODIFIER
from generate_shuffled import compute_derangements
from manifest_io import read_manifest, atomic_write_manifest, acquire_lock, DEFAULT_MANIFEST_PATH

ZERO_VARIANCE_EPS = 1e-8
ARMS = ["constant", "grounded", "shuffled"]

_DERANGEMENTS = compute_derangements(SKY_MODIFIER, GROUND_MODIFIER, VEGETATION_MODIFIER)


def load_rgb01(path: Path) -> np.ndarray:
    """Inverse of fog_utils.save_image: uint8 PNG -> float32 in [0,1]."""
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def is_zero_variance(arr: np.ndarray) -> bool:
    return bool(arr.max() - arr.min() < ZERO_VARIANCE_EPS)


def compute_grounding_depth_corr(beta_map: np.ndarray, depth: np.ndarray) -> float:
    if is_zero_variance(beta_map):
        return 0.0
    return float(pearsonr(beta_map.ravel(), depth.ravel())[0])


def compute_grounding_ref_corr(beta_map: np.ndarray, reference_beta_map: np.ndarray) -> float:
    if is_zero_variance(beta_map) or is_zero_variance(reference_beta_map):
        return 0.0
    return float(pearsonr(beta_map.ravel(), reference_beta_map.ravel())[0])


def compute_arm_stats(foggy_rgb: np.ndarray, clean_rgb: np.ndarray, beta_map: np.ndarray,
                       reference_beta_map: np.ndarray, depth: np.ndarray, aux_path: Path) -> dict:
    mean_abs_delta = float(np.mean(np.abs(foggy_rgb - clean_rgb)))

    aux = np.load(aux_path)["aux"]
    mean_transmission = float(np.mean(1.0 - aux[:, :, 0].astype(np.float32)))

    return {
        "mean_abs_delta": mean_abs_delta,
        "grounding_depth_corr": compute_grounding_depth_corr(beta_map, depth),
        "grounding_ref_corr": compute_grounding_ref_corr(beta_map, reference_beta_map),
        "mean_transmission": mean_transmission,
    }


def build_beta_map_for_arm(arm: str, seg: np.ndarray, depth: np.ndarray, beta_base: float, shuffle_seed) -> np.ndarray:
    if arm == "constant":
        return np.full(depth.shape, beta_base, dtype=np.float32)
    if arm == "grounded":
        return build_beta_map(seg, depth, beta_base, DEFAULT_SIGMA)
    if arm == "shuffled":
        if shuffle_seed is None:
            raise SystemExit("fog.shuffle_seed is null -- run scripts/populate_shuffle_seeds.py first.")
        permutation = random.Random(shuffle_seed).choice(_DERANGEMENTS)
        return build_beta_map(
            seg, depth, beta_base, DEFAULT_SIGMA,
            sky_mod=permutation["sky"], ground_mod=permutation["ground"], veg_mod=permutation["veg"],
        )
    raise ValueError(f"unknown arm: {arm}")


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
    n_arm_not_ready = 0

    for i, entry in enumerate(entries, 1):
        split, city, image = entry["split"], entry["city"], entry["image"]

        arms_needed = [arm for arm in ARMS if entry["fog"][arm]["stats"]["mean_abs_delta"] is None]
        if not arms_needed:
            n_skipped += 1
            continue

        if i % 100 == 0 or i == 1:
            print(f"[{i}/{len(entries)}] {city}/{image}  (arms: {arms_needed})")

        beta_base = entry["fog"]["beta_base"]
        shuffle_seed = entry["fog"]["shuffle_seed"]

        clean = load_clean(split, city, image)
        disparity = load_disparity(split, city, image)
        depth = disparity_to_pseudo_depth(disparity)
        seg = load_seg_labels(split, city, image)

        if seg.shape != depth.shape:
            print(f"  WARNING: shape mismatch (seg={seg.shape}, depth={depth.shape}) -- skipping {city}/{image}")
            continue

        # canonical reference, computed once per image regardless of which
        # arms are being processed -- reused as the correlation target for
        # every arm (including grounded itself, which reuses the identical
        # array and so correlates at ~1.0 with it trivially).
        reference_beta_map = build_beta_map(seg, depth, beta_base, DEFAULT_SIGMA)

        any_computed = False
        for arm in arms_needed:
            rgb_path = PROJECT_ROOT / entry["file_paths"]["fog"][arm]["rgb"]
            aux_path = PROJECT_ROOT / entry["file_paths"]["fog"][arm]["aux"]

            if not rgb_path.exists() or not aux_path.exists():
                # arm not generated yet (e.g. shuffled, before its batch
                # generation has run) -- leave null, not an error.
                n_arm_not_ready += 1
                continue

            foggy = load_rgb01(rgb_path)
            beta_map = reference_beta_map if arm == "grounded" else build_beta_map_for_arm(arm, seg, depth, beta_base, shuffle_seed)
            stats = compute_arm_stats(foggy, clean, beta_map, reference_beta_map, depth, aux_path)
            entry["fog"][arm]["stats"] = stats
            any_computed = True

            if args.limit:
                print(f"  {arm}: {stats}")

        if any_computed:
            n_processed += 1
        else:
            n_skipped += 1

    print(f"\nProcessed {n_processed}, skipped {n_skipped} (all needed arms already done or not ready), "
          f"of {len(entries)} considered. ({n_arm_not_ready} individual arm-skips for not-yet-generated files.)")

    if n_processed == 0:
        print("Nothing new to write -- manifest unchanged.")
        return

    with acquire_lock(DEFAULT_MANIFEST_PATH) as token:
        current = read_manifest(DEFAULT_MANIFEST_PATH)
        current_by_key = {(e["split"], e["city"], e["image"]): e for e in current}
        n_merged = 0
        for entry in entries:
            key = (entry["split"], entry["city"], entry["image"])
            if key not in current_by_key:
                continue
            merged_any = False
            for arm in ARMS:
                if entry["fog"][arm]["stats"]["mean_abs_delta"] is not None:
                    current_by_key[key]["fog"][arm]["stats"] = entry["fog"][arm]["stats"]
                    merged_any = True
            if merged_any:
                n_merged += 1
        atomic_write_manifest(current, DEFAULT_MANIFEST_PATH, lock_token=token)

    print(f"Manifest updated: {DEFAULT_MANIFEST_PATH} ({n_merged} entries merged in)")


if __name__ == "__main__":
    main()
