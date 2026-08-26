"""
FID-based parameter calibration for scene-grounded fog category modifiers.

Generates fog on a fixed set of ~50 diverse Cityscapes images (see
select_fid_images.py) under several (sky_mod, road_mod, veg_mod, sigma)
combinations, and scores each against real non-homogeneous fog (NH-HAZE
*_hazy.png) via FID. All variants use the SAME 50 images and SAME per-image
beta_base (from the manifest), depth-scaled modifiers (never --flat-modifiers),
and no turbulence — so FID differences are attributable only to the category
modifiers and sigma being swept.

Loads the InceptionV3 model and computes the NH-HAZE reference statistics
ONCE, reusing both across all variants (calling pytorch_fid's CLI per variant
would reload the model and recompute the fixed reference stats every time).

Usage:
  python3 scripts/select_fid_images.py --n 50 --seed 42   # run once first
  python3 scripts/fid_calibration.py
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from extract_depth import get_device, get_pipeline, ensure_depth
from fog_utils import PROJECT_ROOT, apply_asm, disparity_to_pseudo_depth, load_clean, load_seg_labels, save_image
from generate_grounded import build_beta_map

MANIFEST_PATH = PROJECT_ROOT / "data" / "fid_calibration" / "image_manifest.json"
CALIB_ROOT = PROJECT_ROOT / "data" / "fid_calibration"
NH_HAZE_DIR = PROJECT_ROOT / "data" / "evaluation" / "nh_haze"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

# name, sky_mod, road_mod, veg_mod, sigma
VARIANTS = [
    ("current",             0.3, 1.3, 1.1, 15),
    ("sky_lower",           0.2, 1.3, 1.1, 15),
    ("sky_higher",          0.4, 1.3, 1.1, 15),
    ("road_lower",          0.3, 1.2, 1.1, 15),
    ("road_higher",         0.3, 1.5, 1.1, 15),
    ("veg_lower",           0.3, 1.3, 1.0, 15),
    ("veg_higher",          0.3, 1.3, 1.2, 15),
    ("sigma_lower",         0.3, 1.3, 1.1, 10),
    ("sigma_higher",        0.3, 1.3, 1.1, 25),
    ("combined_best_guess", 0.2, 1.5, 1.1, 20),
]


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"No manifest at {MANIFEST_PATH}.\n"
            f"Run first: python3 scripts/select_fid_images.py --n 50 --seed 42"
        )
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


def load_and_cache_images(manifest):
    """Load clean/depth/seg once per image, reused across all variants."""
    device = get_device()
    print(f"Depth device: {device}")
    depth_pipe = get_pipeline(device)

    cache = []
    for i, entry in enumerate(manifest):
        split, city, image = entry["split"], entry["city"], entry["image"]
        print(f"  [{i+1}/{len(manifest)}] loading {city}/{image} ...", end=" ", flush=True)
        clean = load_clean(split, city, image)
        disparity = ensure_depth(depth_pipe, split, city, image)
        depth = disparity_to_pseudo_depth(disparity)
        seg = load_seg_labels(split, city, image)
        if seg.shape != depth.shape:
            raise SystemExit(f"Shape mismatch for {city}/{image}: seg {seg.shape} vs depth {depth.shape}")
        cache.append({
            "city": city, "image": image, "beta_base": entry["beta_base"],
            "clean": clean, "depth": depth, "seg": seg,
        })
        print("ok")

    del depth_pipe  # free the depth model before loading InceptionV3
    return cache


def generate_variant(cache, name, sky_mod, road_mod, veg_mod, sigma):
    out_dir = CALIB_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in cache:
        beta_map = build_beta_map(
            item["seg"], item["depth"], item["beta_base"], sigma,
            flat_modifiers=False, sky_mod=sky_mod, ground_mod=road_mod, veg_mod=veg_mod,
        )
        foggy = apply_asm(item["clean"], item["depth"], beta_map)
        save_image(foggy, out_dir / f"{item['city']}_{item['image']}.png")
    return sorted(out_dir.glob("*.png"))


def get_inception_model(device):
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    try:
        model = model.to(device)
        # smoke-test on this device so we fall back to cpu before wasting a sweep
        with torch.no_grad():
            model(torch.zeros(1, 3, 299, 299, device=device))
        return model, device
    except Exception as e:
        print(f"  (Inception failed on device={device} ({e}); falling back to cpu)")
        return InceptionV3([block_idx]).to("cpu"), "cpu"


def fid_stats(files, model, device):
    from pytorch_fid.fid_score import calculate_activation_statistics
    files = [str(f) for f in files]
    return calculate_activation_statistics(files, model, batch_size=min(50, len(files)), dims=2048, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="override device for Inception (cpu/mps/cuda); auto-detected if omitted")
    args = ap.parse_args()

    manifest = load_manifest()
    print(f"Loaded manifest: {len(manifest)} images\n")

    print("=== Caching clean/depth/seg for all calibration images ===")
    t0 = time.time()
    cache = load_and_cache_images(manifest)
    print(f"Cached {len(cache)} images in {time.time()-t0:.1f}s\n")

    nh_files = sorted(NH_HAZE_DIR.glob("*_hazy.png"))
    if not nh_files:
        raise SystemExit(f"No *_hazy.png files found in {NH_HAZE_DIR}")
    print(f"NH-HAZE reference: {len(nh_files)} images")

    from pytorch_fid.fid_score import calculate_frechet_distance

    device = args.device or get_device()
    print(f"Inception device: {device} (will fall back to cpu automatically if unsupported)")
    model, device = get_inception_model(device)

    print("Computing NH-HAZE reference activation statistics (once)...")
    t0 = time.time()
    mu_ref, sigma_ref = fid_stats(nh_files, model, device)
    print(f"  done in {time.time()-t0:.1f}s\n")

    results = []
    for name, sky_mod, road_mod, veg_mod, sigma in VARIANTS:
        t0 = time.time()
        print(f"=== Variant: {name} (sky={sky_mod}, road={road_mod}, veg={veg_mod}, sigma={sigma}) ===")
        files = generate_variant(cache, name, sky_mod, road_mod, veg_mod, sigma)
        mu, sig = fid_stats(files, model, device)
        fid = calculate_frechet_distance(mu_ref, sigma_ref, mu, sig)
        elapsed = time.time() - t0
        print(f"  FID = {fid:.3f}  ({elapsed:.1f}s)\n")
        results.append({
            "variant": name, "sky_mod": sky_mod, "road_mod": road_mod,
            "veg_mod": veg_mod, "sigma": sigma, "fid": fid,
        })

    results.sort(key=lambda r: r["fid"])

    print("\n" + "=" * 78)
    print("FINAL RESULTS (sorted by FID, lowest = closest to real NH-HAZE fog)")
    print("=" * 78)
    header = f"{'variant':<22}{'sky':>7}{'road':>7}{'veg':>7}{'sigma':>7}{'FID':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['variant']:<22}{r['sky_mod']:>7}{r['road_mod']:>7}{r['veg_mod']:>7}{r['sigma']:>7}{r['fid']:>10.3f}")

    best = results[0]
    print(f"\nWinner: {best['variant']} (FID={best['fid']:.3f}) — "
          f"sky={best['sky_mod']}, road={best['road_mod']}, veg={best['veg_mod']}, sigma={best['sigma']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "fid_calibration_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    csv_path = RESULTS_DIR / "fid_calibration_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["variant", "sky_mod", "road_mod", "veg_mod", "sigma", "fid"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
