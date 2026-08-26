"""
Scan Cityscapes train-split segmentation labels for images with both:
  - a good amount of open sky (upper part of frame), and
  - a road that stretches far into the distance (reaches high up the frame,
    i.e. close to the horizon line rather than only filling the bottom).

These are the images where the scene-grounded beta modifiers (sky x0.3,
road x1.3) will produce the most visually legible difference from
constant-beta fog, since both depend on having real depth range to work
with (a frame dominated by a nearby building has almost no far-field pixels
for the modifier to act on).

Usage:
  python3 scripts/scan_candidates.py --city strasbourg --top 5
  python3 scripts/scan_candidates.py --city strasbourg hamburg zurich --top 3
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEG_ROOT = PROJECT_ROOT / "data" / "segmentation" / "cityscapes"

SKY_ID = 23
ROAD_ID = 7


def score_image(seg: np.ndarray) -> dict:
    h, w = seg.shape
    upper = seg[: int(h * 0.4), :]
    sky_frac_upper = float((upper == SKY_ID).mean())

    road_mask = seg == ROAD_ID
    road_frac = float(road_mask.mean())

    if road_frac < 0.08:
        topmost_road_frac = 1.0  # no meaningful road -> penalize
    else:
        rows_with_road = np.where(road_mask.any(axis=1))[0]
        topmost_road_frac = float(rows_with_road.min()) / h  # smaller = road reaches farther/higher

    score = sky_frac_upper + (1.0 - topmost_road_frac)
    return {
        "sky_frac_upper": sky_frac_upper,
        "road_frac": road_frac,
        "topmost_road_frac": topmost_road_frac,
        "score": score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    for city in args.city:
        city_dir = SEG_ROOT / args.split / city
        label_files = sorted(city_dir.glob("*_labelIds.png"))
        print(f"\n=== {city} ({len(label_files)} images) ===")

        results = []
        for f in label_files:
            seg = np.array(Image.open(f))
            stats = score_image(seg)
            stem = f.stem.replace("_gtFine_labelIds", "")
            results.append((stem, stats))

        results.sort(key=lambda x: x[1]["score"], reverse=True)

        for stem, stats in results[: args.top]:
            print(
                f"  {stem}: score={stats['score']:.3f} "
                f"sky_upper={stats['sky_frac_upper']:.3f} "
                f"road_frac={stats['road_frac']:.3f} "
                f"road_topmost={stats['topmost_road_frac']:.3f}"
            )


if __name__ == "__main__":
    main()
