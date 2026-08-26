"""
Build a comparison grid for the scene-grounded method with vs. without
turbulence: no-turbulence | mild turbulence | aggressive turbulence, fog
output on top and the corresponding beta map underneath each.

Assumes the three grounded variants already exist (generate via
generate_grounded.py with/without --turbulence).

Usage:
  python3 scripts/make_turbulence_comparison.py --split train --city strasbourg \
      --image strasbourg_000001_043748 --beta-base 1.0 \
      --variant-b-strength 0.15 --variant-b-scale 20 \
      --variant-c-strength 0.25 --variant-c-scale 30
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
    ap.add_argument("--variant-b-strength", type=float, default=0.15)
    ap.add_argument("--variant-b-scale", type=float, default=20.0)
    ap.add_argument("--variant-c-strength", type=float, default=0.25)
    ap.add_argument("--variant-c-scale", type=float, default=30.0)
    args = ap.parse_args()

    B = args.beta_base
    img_dir = GROUNDED_ROOT / args.split / args.city

    stem_a = f"{args.image}_betabase{B:.2f}_grounded"
    stem_b = f"{stem_a}_turb{args.variant_b_strength:.2f}-{args.variant_b_scale:.0f}"
    stem_c = f"{stem_a}_turb{args.variant_c_strength:.2f}-{args.variant_c_scale:.0f}"

    panels = [
        (img_dir / f"{stem_a}.png", "no turbulence"),
        (img_dir / f"{stem_b}.png", f"turbulence strength={args.variant_b_strength}, scale={args.variant_b_scale:.0f}"),
        (img_dir / f"{stem_c}.png", f"turbulence strength={args.variant_c_strength}, scale={args.variant_c_scale:.0f} (aggressive)"),
        (img_dir / f"{stem_a}_betamap.png", "beta map: no turbulence"),
        (img_dir / f"{stem_b}_betamap.png", "beta map: mild turbulence"),
        (img_dir / f"{stem_c}_betamap.png", "beta map: aggressive turbulence"),
    ]

    for path, _ in panels:
        if not path.exists():
            raise SystemExit(f"Missing file: {path}\nGenerate it first via generate_grounded.py")

    labeled = [label_panel(load_and_resize(p), text) for p, text in panels]
    grid = make_grid(labeled, cols=3)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{args.city}_{args.image}_beta{B:.1f}_turbulence_compare.png"
    grid.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
