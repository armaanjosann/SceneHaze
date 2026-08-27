"""
Generate the MCBM (HazeFlow) fog baseline, using HazeFlow's PUBLISHED
formula (Shin et al., Eq. 22 in the paper):

  T_MCBM(x) = exp( -(beta + alpha * beta_tilde(x)) * d(x) )
  I(x) = T_MCBM(x) * J(x) + (1 - T_MCBM(x)) * A

  where beta ~ Uniform(0.2, 2.8)   -- their scalar draw (we substitute our
                                      own per-image beta_base instead, for
                                      comparability with our other methods)
        alpha ~ Uniform(0.5, 1.0)  -- controls degree of non-homogeneity
        beta_tilde(x)              -- their Brownian-motion field (datasets/
                                      MCBM/*.png), normalized to [0,1]

IMPORTANT CORRECTION: an earlier version of this script copied a
`depth * 2.0 * beta` scaling from HazeFlow's GitHub code
(reflow/datasets.py::MCBM), on the assumption that the repo was the more
literal ground truth. Reading the actual paper (Eq. 22) shows no such
constant -- the code's extra `2.0` isn't part of the published equation
and is most likely an artifact tuned to their specific depth estimator
(RA-Depth) and dataset (RIDCP500), which we aren't using anyway (we use
Cityscapes + Depth-Anything-V2). Blindly carrying that constant over onto
a different depth estimator's output range was the more likely source of
a highly aggressive fog when this script's original results looked too
dense. This version follows the paper's equation exactly, with no extra
depth scaling.

Known, deliberate deviations from the paper, for comparability with our
other two baselines (documented so this can be cited accurately):
  - beta_base is OUR per-image value (from mcbm_kaya_selection.json),
    substituted for their own random per-sample Uniform(0.2, 2.8) draw.
  - Atmospheric light A is fixed at our own convention
    (fog_utils.ATMOSPHERIC_LIGHT, near-white) rather than their randomized
    A ~ Uniform(0.25, 1.8) sampling.
  - We do NOT apply their Eq. 23 degradation step D(...) (gamma
    correction + additive Gaussian noise + JPEG compression) -- this
    affects sensor/compression grain, not the fog's spatial structure,
    but is a real omission if full fidelity to their training data is
    the goal.
  - Clean images are Cityscapes, not their RIDCP500 set; depth comes from
    Depth-Anything-V2, not their RA-Depth. Both were already-documented
    substitutions from earlier in this project (RIDCP500's images were
    only reachable via an inaccessible Baidu Disk link).

No GPU needed -- this is plain numpy/PIL image arithmetic, no model
inference involved.

Expects:
  <input-root>/clean/<split>/<city>/<image>_leftImg8bit.png
  <input-root>/depth/<split>/<city>/<image>_depth.npy
  <input-root>/mcbm_kaya_selection.json
  <hazeflow-root>/datasets/MCBM/*.png   (from a plain `git clone` of HazeFlow
                                         -- these ship precomputed in the repo)

Usage:
  python3 scripts/generate_mcbm_kaya.py \
      --input-root ~/scenehaze_work/mcbm_kaya_transfer \
      --hazeflow-root ~/scenehaze_work/HazeFlow \
      --output-root ~/mcbm_output \
      --seed 123
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from fog_utils import ATMOSPHERIC_LIGHT, disparity_to_pseudo_depth, save_image

SCALE_RANGE = (0.5, 1.0)  # matches HazeFlow's own alpha ~ Uniform(0.5, 1.0)


def load_mcbm_field(hazeflow_root: Path, index: int, shape) -> np.ndarray:
    path = hazeflow_root / "datasets" / "MCBM" / f"{index}.png"
    if not path.exists():
        raise SystemExit(f"Missing MCBM field: {path}\n(did you `git clone https://github.com/cloor/HazeFlow.git` here?)")
    nh = Image.open(path).convert("L").resize((shape[1], shape[0]), Image.BICUBIC)
    nh = np.array(nh).astype(np.float32) / 255.0
    return (nh - nh.min()) / (nh.max() - nh.min() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path.home() / "mcbm_kaya_transfer")
    ap.add_argument("--hazeflow-root", type=Path, default=Path.home() / "HazeFlow")
    ap.add_argument("--output-root", type=Path, default=Path.home() / "mcbm_output")
    ap.add_argument("--seed", type=int, default=123, help="seed for per-image mcbm-index/scale draws")
    args = ap.parse_args()

    manifest_path = args.input_root / "mcbm_kaya_selection.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    selection = json.load(open(manifest_path))
    print(f"Loaded {len(selection)} images from {manifest_path}")

    rng = random.Random(args.seed)

    for i, e in enumerate(selection, 1):
        split, city, image, beta_base = e["split"], e["city"], e["image"], e["beta_base"]
        print(f"[{i}/{len(selection)}] {city}/{image} (beta_base={beta_base:.4f})")

        clean_path = args.input_root / "clean" / split / city / f"{image}_leftImg8bit.png"
        depth_path = args.input_root / "depth" / split / city / f"{image}_depth.npy"

        clean = np.array(Image.open(clean_path).convert("RGB")).astype(np.float32) / 255.0
        disparity = np.load(depth_path)
        depth = disparity_to_pseudo_depth(disparity)

        mcbm_index = rng.randint(0, 999)
        scale = rng.uniform(*SCALE_RANGE)
        nh = load_mcbm_field(args.hazeflow_root, mcbm_index, depth.shape)

        beta_map = nh * scale + beta_base  # beta + alpha*beta_tilde, per Eq. 22
        t = np.exp(-depth * beta_map)      # T_MCBM, per Eq. 22 (no extra depth scaling)
        hazy = np.clip(clean * t[:, :, None] + ATMOSPHERIC_LIGHT * (1 - t[:, :, None]), 0, 1)

        out_dir = args.output_root / split / city
        stem = f"{image}_betabase{beta_base:.2f}_mcbm"
        save_image(hazy, out_dir / f"{stem}.png")

        beta_vis = (beta_map - beta_map.min()) / (beta_map.max() - beta_map.min() + 1e-8)
        save_image(np.repeat(beta_vis[:, :, None], 3, axis=2), out_dir / f"{stem}_betamap.png")

        print(f"  mcbm_index={mcbm_index}, scale={scale:.3f}, "
              f"beta_map range=[{beta_map.min():.3f}, {beta_map.max():.3f}]")

    print(f"\nDone. {len(selection)} images written to {args.output_root}")


if __name__ == "__main__":
    main()
