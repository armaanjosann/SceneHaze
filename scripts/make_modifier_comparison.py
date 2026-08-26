"""
Build a comparison grid for scene-grounded fog with flat vs. depth-scaled
category modifiers: fog output on top, beta map underneath, for each.

Assumes both variants already exist (generate via generate_grounded.py,
with/without --flat-modifiers).

Usage:
  python3 scripts/make_modifier_comparison.py --split train --city strasbourg \
      --image strasbourg_000001_043748 --beta-base 1.0
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
    args = ap.parse_args()

    B = args.beta_base
    img_dir = GROUNDED_ROOT / args.split / args.city
    base_stem = f"{args.image}_betabase{B:.2f}_grounded"
    flat_stem = f"{base_stem}_flatmod"

    panels = [
        (img_dir / f"{flat_stem}.png", "flat modifiers (ablation): road always x1.3"),
        (img_dir / f"{base_stem}.png", "depth-scaled modifiers (default): road x(1.0 + 0.3*depth)"),
        (img_dir / f"{flat_stem}_betamap.png", "beta map: flat"),
        (img_dir / f"{base_stem}_betamap.png", "beta map: depth-scaled"),
    ]

    for path, _ in panels:
        if not path.exists():
            raise SystemExit(f"Missing file: {path}\nGenerate it first via generate_grounded.py")

    labeled = [label_panel(load_and_resize(p), text) for p, text in panels]
    grid = make_grid(labeled, cols=2)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{args.city}_{args.image}_beta{B:.1f}_modifier_compare.png"
    grid.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
