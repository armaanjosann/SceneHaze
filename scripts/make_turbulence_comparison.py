"""
Build a comparison grid for the scene-grounded method across any number of
turbulence settings: no-turbulence baseline plus one or more
(strength, scale) variants, fog output on top and its beta map underneath.

Assumes all variants already exist (generate via generate_grounded.py,
with/without --turbulence).

Usage:
  python3 scripts/make_turbulence_comparison.py --split train --city strasbourg \
      --image strasbourg_000001_043748 --beta-base 1.0 \
      --variant 0.05 50 --variant 0.08 60 --variant 0.12 80
"""

import argparse

from fog_utils import PROJECT_ROOT, load_and_resize, label_panel, make_grid

GROUNDED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded"
OUT_ROOT = PROJECT_ROOT / "results" / "comparisons"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", required=True)
    ap.add_argument("--image", required=True, help="image stem, no suffix")
    ap.add_argument("--beta-base", type=float, default=1.0)
    ap.add_argument(
        "--variant", nargs=2, type=float, action="append", metavar=("STRENGTH", "SCALE"),
        default=None,
        help="turbulence (strength, scale) pair; repeatable. Defaults to three gentle settings if omitted.",
    )
    args = ap.parse_args()

    variants = args.variant or [(0.05, 50.0), (0.08, 60.0), (0.12, 80.0)]

    B = args.beta_base
    img_dir = GROUNDED_ROOT / args.split / args.city
    base_stem = f"{args.image}_betabase{B:.2f}_grounded"

    panels_fog = [(img_dir / f"{base_stem}.png", "no turbulence")]
    panels_beta = [(img_dir / f"{base_stem}_betamap.png", "beta map: no turbulence")]

    for strength, scale in variants:
        stem = f"{base_stem}_turb{strength:.2f}-{scale:.0f}"
        label = f"turbulence strength={strength}, scale={scale:.0f}"
        panels_fog.append((img_dir / f"{stem}.png", label))
        panels_beta.append((img_dir / f"{stem}_betamap.png", f"beta map: {label}"))

    panels = panels_fog + panels_beta
    for path, _ in panels:
        if not path.exists():
            raise SystemExit(f"Missing file: {path}\nGenerate it first via generate_grounded.py")

    labeled = [label_panel(load_and_resize(p), text) for p, text in panels]
    grid = make_grid(labeled, cols=len(panels_fog))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{args.city}_{args.image}_beta{B:.1f}_turbulence_compare.png"
    grid.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
