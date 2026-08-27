"""
Pick a few images with strong sky+road+vegetation diversity (scored via
scan_candidates.score_image) and build constant-vs-grounded comparison
figures (+ beta map) using the ALREADY-GENERATED dataset outputs at each
image's manifest-assigned beta_base -- no new fog generation.

Also prints the mean absolute pixel difference between constant and
grounded per image: a number to back up (or contradict) the visual
impression of whether scene-grounding is a meaningful effect or too subtle.

Usage:
  python3 scripts/make_effect_comparison.py --n 3
"""

import argparse
import json

import numpy as np
from PIL import Image

from fog_utils import PROJECT_ROOT, SEG_ROOT, load_and_resize, label_panel, make_grid
from scan_candidates import score_image
from generate_grounded import SKY_IDS, GROUND_IDS, VEGETATION_IDS

MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"
CONSTANT_ROOT = PROJECT_ROOT / "data" / "generated" / "constant_beta"
GROUNDED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded"
OUT_ROOT = PROJECT_ROOT / "results" / "comparisons"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST_PATH))
    print(f"Scoring {len(manifest)} images for sky+road+vegetation diversity...")

    scored = []
    for e in manifest:
        seg_path = SEG_ROOT / e["split"] / e["city"] / f"{e['image']}_gtFine_labelIds.png"
        seg = np.array(Image.open(seg_path))
        stats = score_image(seg)
        scored.append((stats["score"], stats, e))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top N from DIFFERENT cities, so the sample isn't three near-identical
    # frames from the same driving sequence.
    picked = []
    seen_cities = set()
    for score, stats, e in scored:
        if e["city"] in seen_cities:
            continue
        picked.append((score, stats, e))
        seen_cities.add(e["city"])
        if len(picked) >= args.n:
            break

    print(f"\nPicked {len(picked)} images:")
    for score, stats, e in picked:
        print(f"  {e['city']}/{e['image']}  score={score:.3f}  "
              f"sky_upper={stats['sky_frac_upper']:.3f}  veg_frac={stats['veg_frac']:.3f}  "
              f"beta_base={e['beta_base']:.2f}")

    print()
    for score, stats, e in picked:
        split, city, image, beta = e["split"], e["city"], e["image"], e["beta_base"]

        constant_path = CONSTANT_ROOT / split / city / f"{image}_beta{beta:.2f}_constant.png"
        grounded_stem = f"{image}_betabase{beta:.2f}_grounded"
        grounded_path = GROUNDED_ROOT / split / city / f"{grounded_stem}.png"
        betamap_path = GROUNDED_ROOT / split / city / f"{grounded_stem}_betamap.png"

        for p in (constant_path, grounded_path, betamap_path):
            if not p.exists():
                raise SystemExit(f"Missing expected dataset output: {p}")

        const_arr = np.array(Image.open(constant_path).convert("RGB")).astype(np.float32)
        ground_arr = np.array(Image.open(grounded_path).convert("RGB")).astype(np.float32)
        abs_diff = np.abs(const_arr - ground_arr).mean(axis=2)  # per-pixel, averaged over RGB
        mad = float(abs_diff.mean())

        seg_path = SEG_ROOT / split / city / f"{image}_gtFine_labelIds.png"
        seg = np.array(Image.open(seg_path))
        sky_mask = np.isin(seg, list(SKY_IDS))
        road_mask = np.isin(seg, list(GROUND_IDS))
        veg_mask = np.isin(seg, list(VEGETATION_IDS))
        other_mask = ~(sky_mask | road_mask | veg_mask)

        def masked_mad(mask):
            return float(abs_diff[mask].mean()) if mask.any() else float("nan")

        cat_mad = {
            "sky": (masked_mad(sky_mask), sky_mask.mean()),
            "road": (masked_mad(road_mask), road_mask.mean()),
            "vegetation": (masked_mad(veg_mask), veg_mask.mean()),
            "everything else": (masked_mad(other_mask), other_mask.mean()),
        }

        panels = [
            (constant_path, f"constant beta={beta:.2f} (baseline)"),
            (grounded_path, f"scene-grounded beta_base={beta:.2f} (ours)"),
            (betamap_path, "beta map"),
        ]
        labeled = [label_panel(load_and_resize(p), text) for p, text in panels]
        grid = make_grid(labeled, cols=3)

        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = OUT_ROOT / f"{city}_{image}_effect_compare.png"
        grid.save(out_path)

        print(f"{city}/{image}  (beta_base={beta:.2f}, diversity score={score:.3f})")
        print(f"  Whole-image mean absolute pixel difference: {mad:.3f} / 255  ({mad/255*100:.2f}%)")
        print(f"  By category (mean abs diff / 255, % of frame):")
        for cat, (cmad, frac) in cat_mad.items():
            print(f"    {cat:<16} {cmad:6.3f}  ({frac*100:5.1f}% of frame)")
        print(f"  Saved: {out_path}\n")


if __name__ == "__main__":
    main()
