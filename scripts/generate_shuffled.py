"""
Piece 3 (fog) — shuffled-modifier ablation. Uses the exact same
build_beta_map machinery as scene-grounded fog, but the three category
modifiers [sky=0.2, ground=1.5, veg=1.1] get randomly reassigned to a
DIFFERENT category than their own, per image -- e.g. sky might get 1.5
(ground's modifier) while ground gets 1.1 (veg's) and veg gets 0.2
(sky's). This preserves everything about the grounded pipeline (same
category masks, same depth-scaling, same Gaussian smoothing, same
apply_asm) except which modifier value lands on which category -- testing
whether the SPATIAL ALIGNMENT of modifiers to scene semantics is what
matters, not just having per-pixel variance somewhere.

Permutation is a uniform random DERANGEMENT (no category ever keeps its
own original modifier), not a uniform-over-all-6-permutations draw -- a
draw that included the identity permutation would silently make that
image's shuffled output pixel-identical to grounded for ~1/6 of images,
diluting the ablation for no benefit. For 3 categories there are exactly
2 derangements (the two 3-cycles); computed generically from whatever
SKY_MODIFIER/GROUND_MODIFIER/VEGETATION_MODIFIER currently are, not
hardcoded, so this stays correct if those values are ever recalibrated.

The permutation is deterministic per image: derived from
sha256(f"{image}_{global_seed}") -- NOT Python's built-in hash(), which is
randomized per-process by default (PYTHONHASHSEED) and would silently
break reproducibility across runs/machines.

IMPORTANT, DO NOT "FIX": this arm is NOT energy-matched to grounded on a
per-image basis, and that's expected, not a bug. With only 2 categories
(sky, ground) that structurally interact with depth-scaling asymmetrically
(sky flat, ground/veg depth-scaled) and only 3 categories total, there are
exactly 2 valid derangements -- meaning sky NEVER keeps its protective low
modifier (0.2) in ANY shuffled output, and ground NEVER keeps its boost
(1.5). So the size and direction of each image's total fog-energy shift
(measured as mean_abs_delta in the manifest) is systematically tied to
that image's own sky/ground/veg composition and which of the 2
derangements got drawn, not randomly scattered around grounded's value.
Measured across all 3,475 images: dataset-wide mean_abs_delta ratio
(shuffled/grounded) is ~0.85 (close to a naive "similar energy"
expectation), but the PER-IMAGE ratio ranges from ~0.50 to ~1.09 -- half
the dataset falls outside a +/-15% band around 1.0. This means shuffled is
a semantic-scrambling control, not an energy-matched one: any
grounded > shuffled result on a downstream metric should be read as a
LOWER BOUND on the semantic-correctness effect (since shuffled's energy
sometimes exceeds grounded's for sky-light/ground-heavy images), not a
clean isolation of "semantics held constant, only alignment changed."
Do not try to "fix" this by adding more derangement options or reweighting
without discussing first -- it's a structural property of having only 3
categories, and changing it changes what the ablation is measuring.

Usage:
  python3 scripts/generate_shuffled.py \
      --image aachen_000000_000019 --split train --city aachen --beta-base 1.0
"""

import argparse
import hashlib
import itertools
import random

import numpy as np

from fog_utils import (
    PROJECT_ROOT,
    apply_asm,
    disparity_to_pseudo_depth,
    load_clean,
    load_disparity,
    load_seg_labels,
    save_aux,
    save_image,
)
from generate_grounded import build_beta_map, DEFAULT_SIGMA, SKY_MODIFIER, GROUND_MODIFIER, VEGETATION_MODIFIER

OUT_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded_shuffled"

GLOBAL_SEED = 42  # same global seed as beta_base's assignment (generate_dataset.py)


def derive_shuffle_seed(image: str, global_seed: int = GLOBAL_SEED) -> int:
    """Deterministic per-image seed, stable across processes/machines
    (unlike Python's built-in hash(), which is randomized per-process by
    default)."""
    key = f"{image}_{global_seed}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big") % (2 ** 31)


def compute_derangements(sky_mod: float, ground_mod: float, veg_mod: float) -> list:
    """All permutations of the three modifier values across the three
    categories where no category keeps its own original value."""
    original = {"sky": sky_mod, "ground": ground_mod, "veg": veg_mod}
    categories = ["sky", "ground", "veg"]
    values = [sky_mod, ground_mod, veg_mod]
    derangements = []
    for perm in itertools.permutations(values):
        candidate = dict(zip(categories, perm))
        if all(candidate[c] != original[c] for c in categories):
            derangements.append(candidate)
    return derangements


def pick_permutation(image: str, global_seed: int = GLOBAL_SEED) -> dict:
    seed = derive_shuffle_seed(image, global_seed)
    derangements = compute_derangements(SKY_MODIFIER, GROUND_MODIFIER, VEGETATION_MODIFIER)
    rng = random.Random(seed)
    return rng.choice(derangements), seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="aachen_000000_000019", help="image stem, no suffix")
    ap.add_argument("--split", default="train")
    ap.add_argument("--city", default="aachen")
    ap.add_argument("--beta-base", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    args = ap.parse_args()

    clean = load_clean(args.split, args.city, args.image)
    disparity = load_disparity(args.split, args.city, args.image)
    depth = disparity_to_pseudo_depth(disparity)
    seg = load_seg_labels(args.split, args.city, args.image)

    if seg.shape != depth.shape:
        raise SystemExit(
            f"Shape mismatch: seg {seg.shape} vs depth {depth.shape} — "
            f"clean/seg/depth must all be the same resolution."
        )

    permutation, shuffle_seed = pick_permutation(args.image)

    beta_map = build_beta_map(
        seg, depth, args.beta_base, args.sigma, flat_modifiers=False,
        sky_mod=permutation["sky"], ground_mod=permutation["ground"], veg_mod=permutation["veg"],
    )

    foggy = apply_asm(clean, depth, beta_map)

    stem = f"{args.image}_betabase{args.beta_base:.2f}_shuffled"
    out_dir = OUT_ROOT / args.split / args.city
    save_image(foggy, out_dir / f"{stem}.png")

    # Bonus debug/thesis-figure visualization -- not part of the manifest's
    # formal file_paths schema (same as grounded's _betamap.png, which also
    # isn't tracked there), just useful for visual review.
    beta_vis = (beta_map - beta_map.min()) / (beta_map.max() - beta_map.min() + 1e-8)
    save_image(np.repeat(beta_vis[:, :, None], 3, axis=2), out_dir / f"{stem}_betamap.png")

    veiling_density = 1.0 - np.exp(-beta_map * depth)
    save_aux(veiling_density, out_dir / f"{stem}_aux.npz")

    print(f"beta_base = {args.beta_base}, sigma = {args.sigma}")
    print(f"shuffle_seed = {shuffle_seed}")
    print(f"permutation: sky_mod={permutation['sky']}, ground_mod={permutation['ground']}, veg_mod={permutation['veg']}")
    print(f"  (originals were: sky={SKY_MODIFIER}, ground={GROUND_MODIFIER}, veg={VEGETATION_MODIFIER})")
    print(f"beta_map range: [{beta_map.min():.3f}, {beta_map.max():.3f}]")
    print(f"Saved: {out_dir / (stem + '.png')}")
    print(f"Saved beta map viz: {out_dir / (stem + '_betamap.png')}")
    print(f"Saved aux: {out_dir / (stem + '_aux.npz')}")


if __name__ == "__main__":
    main()
