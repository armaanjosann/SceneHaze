"""
Select ~N diverse Cityscapes images (spread across multiple train cities,
favoring open sky + long/receding road per scan_candidates.py's scoring) for
FID-based parameter calibration against NH-HAZE.

Also assigns each selected image a fixed, seeded beta_base (Uniform in
--beta-range, default [0.4, 2.0] per the project's severity convention) and
writes everything to a manifest JSON. This is what makes the calibration
sweep fair: every parameter variant generates fog on the *same* images at
the *same* per-image severities — only the category modifiers change.

Usage:
  python3 scripts/select_fid_images.py --n 50 --seed 42
"""

import argparse
import json
import math

import numpy as np
from PIL import Image

from fog_utils import PROJECT_ROOT, SEG_ROOT
from scan_candidates import score_image

TRAIN_ROOT = SEG_ROOT / "train"
OUT_PATH = PROJECT_ROOT / "data" / "fid_calibration" / "image_manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42, help="seed for beta_base assignment")
    ap.add_argument("--beta-range", type=float, nargs=2, default=[0.4, 2.0])
    args = ap.parse_args()

    cities = sorted(p.name for p in TRAIN_ROOT.iterdir() if p.is_dir())
    per_city = math.ceil(args.n / len(cities))

    candidates = []  # (score, city, stem)
    for city in cities:
        label_files = sorted((TRAIN_ROOT / city).glob("*_labelIds.png"))
        scored = []
        for f in label_files:
            seg = np.array(Image.open(f))
            stats = score_image(seg)
            stem = f.stem.replace("_gtFine_labelIds", "")
            scored.append((stats["score"], city, stem))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates.extend(scored[:per_city])
        print(f"  {city}: scanned {len(label_files)}, kept top {min(per_city, len(scored))}")

    # Keep the globally best `n` across all cities' top candidates (drops the
    # weakest few from whichever cities scored generally lower), but assign
    # beta_base in a canonical (city, stem) order so it's independent of any
    # score-based tie-breaking -- fully deterministic given the seed.
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[: args.n]
    selected.sort(key=lambda x: (x[1], x[2]))

    rng = np.random.default_rng(args.seed)
    beta_bases = rng.uniform(args.beta_range[0], args.beta_range[1], size=len(selected))

    manifest = [
        {"split": "train", "city": city, "image": stem, "score": round(score, 4), "beta_base": round(float(beta), 4)}
        for (score, city, stem), beta in zip(selected, beta_bases)
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)

    city_counts = {}
    for m in manifest:
        city_counts[m["city"]] = city_counts.get(m["city"], 0) + 1

    print(f"\nSelected {len(manifest)} images across {len(city_counts)} cities:")
    for city, n in sorted(city_counts.items()):
        print(f"  {city}: {n}")
    print(f"\nSaved manifest: {OUT_PATH}")


if __name__ == "__main__":
    main()
