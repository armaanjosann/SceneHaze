"""
Generate the MCBM (HazeFlow) fog baseline on Kaya, using HazeFlow's actual
formula from their training dataloader (reflow/datasets.py::MCBM), run here
as a standalone generation script rather than inside their training loop.

Unlike our local scripts/generate_mcbm.py (which deliberately keeps our own
ASM equation form t=exp(-beta*depth), for a fair comparison against our other
two methods), THIS script uses HazeFlow's LITERAL formula, including their
hardcoded depth*2.0 scaling -- maximum fidelity to their published code,
since the point of running this on Kaya is "the real baseline," not our
reimplementation of it.

  beta_map(x) = nh(x) * scale + beta_base     (scale ~ Uniform(0.5, 1.0),
                                                matching their own
                                                (rand()+1)/2 term)
  t(x) = exp(-depth(x) * 2.0 * beta_map(x))
  hazy = clean * t + A * (1 - t)

beta_base is OUR per-image value (from mcbm_kaya_selection.json), substituted
for HazeFlow's own random per-sample draw -- so this stays comparable to our
other two baselines, which all use this same beta_base per image. Atmospheric
light A is fixed at our own convention (fog_utils.ATMOSPHERIC_LIGHT, near-
white) rather than HazeFlow's own random A sampling, for the same reason:
every other variable stays identical across all three methods except beta.

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

SCALE_RANGE = (0.5, 1.0)  # matches HazeFlow's own (np.random.rand()+1)/2
DEPTH_SCALE_FACTOR = 2.0  # matches HazeFlow's own t = exp(-depth * 2.0 * beta)


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

        beta_map = nh * scale + beta_base
        t = np.exp(-depth * DEPTH_SCALE_FACTOR * beta_map)
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
