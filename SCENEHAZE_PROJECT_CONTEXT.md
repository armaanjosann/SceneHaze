# SceneHaze: Scene-Grounded Non-Homogeneous Fog Synthesis for Image Restoration Training

## Project Overview

Honours thesis supervised by Saeed Anwar. Goal: generate more realistic synthetic fog images for training image restoration (dehazing) models, and prove that better training data produces better restorers.

---

## The Gap (Verified)

Every major restoration model (NAFNet, PromptIR, Restormer, OneRestore, etc.) trains on RESIDE — which generates fog using the Atmospheric Scattering Model (ASM) with **constant β** (uniform fog density everywhere). Real fog is non-homogeneous: pools in low areas, denser near water, thins toward sky.

**Verified facts:**
- Every restoration model trains on RESIDE's homogeneous fog. Every single one.
- NH-HAZE (2020) proved models fail on real non-homogeneous fog (best PSNR only ~17 dB vs ~21+ dB on homogeneous benchmarks).
- Four NTIRE challenges (2020-2024), hundreds of teams — all build better architectures, none build better training data.
- No prior work uses semantic segmentation to guide fog density placement in ASM for restoration training (verified via Consensus AI and web search).
- No large-scale synthetic non-homogeneous fog dataset has been validated as restoration training data.
- HazeFlow's MCBM (ICCV 2025) varies β randomly via Brownian motion — better than constant, but not scene-aware.

---

## The Contribution

**Contribution 1 — Generation Method:** Modify standard ASM by making β vary spatially based on scene understanding (depth + semantic segmentation). Training-free, closed-form math.

**Contribution 2 — Restoration Validation:** Benchmark 5-6 SOTA restoration models, pick best, fine-tune on our data, show improvement on real non-homogeneous fog benchmarks.

---

## The Core Equation: ASM

```
I(x) = J(x) · t(x) + A · (1 - t(x))
where t(x) = e^(-β · d(x))
```

| Symbol | Meaning |
|--------|---------|
| J(x) | Clean image |
| I(x) | Foggy image (output) |
| d(x) | Depth at pixel x |
| β | Fog density |
| t(x) | Transmission (how much light survives) |
| A | Atmospheric light color (near-white, e.g. [0.85, 0.85, 0.85]) |

**Standard (RESIDE):** β = one constant, A = one constant. Uniform fog.
**HazeFlow MCBM:** β varies randomly (Brownian motion). Patchy but not scene-aware.
**Ours:** β(x) varies based on depth + segmentation. A(x) optionally varies near light sources.

---

## Three-Way Comparison

| Condition | β behavior | Source |
|-----------|-----------|--------|
| Baseline 1: Constant β | Fixed number, uniform | Standard ASM (RESIDE) |
| Baseline 2: Random β(x) | Random Brownian motion | HazeFlow MCBM (ICCV 2025) |
| Ours: Scene-grounded β(x) | Varies by depth + surface type | Our method |

All three use the **same** clean images and **same** depth maps. Only β computation differs.

---

## Scene-Grounded β(x) Logic

Segmentation classes collapsed into 6 fog-relevant categories:

| Fog Category | Example Classes | β Modifier | Reason |
|-------------|----------------|-----------|--------|
| Sky | sky | ×0.3 | Fog thins above atmospheric boundary |
| Ground/road | road, sidewalk | ×1.3 | Fog pools at ground level |
| Water | water, river | ×1.3-1.5 | Moisture source |
| Low vegetation | grass, field | ×1.1 | Ground-level, traps moisture |
| Structures | building, wall | ×1.0 | Vertical surfaces, no pooling |
| Everything else | car, person, etc. | ×1.0 | Default |

**Modifier values are starting points** — tune by visual comparison with NH-HAZE + FID measurement.

After applying modifiers, smooth β_map with `gaussian_filter(beta_map, sigma=15)` to avoid harsh edges.

**Severity variation:** β_base randomly sampled from [0.4, 2.0] per image (light to heavy fog).

---

## Current Setup

### What's Been Downloaded

**Cityscapes clean images** (`leftImg8bit_trainvaltest.zip` — 5,000 images, 2048×1024):
- Located at: `~/Desktop/leftImg8bit_trainvaltest/`
- Structure: `leftImg8bit/{train,val,test}/{city_name}/` with .png images
- Images named like: `aachen_000000_000019_leftImg8bit.png`

**Cityscapes ground truth segmentation** (`gtFine_trainvaltest.zip` — perfect semantic labels):
- Located at: `~/Desktop/gtFine_trainvaltest (1)/`
- Structure: `gtFine/{train,val,test}/{city_name}/` with multiple files per image
- Contains `_labelIds.png`, `_instanceIds.png`, `_color.png`, `_polygons.json` per image
- **Use `_labelIds.png`** — each pixel value is a class ID

**Note on RIDCP500:** Original plan was to use RIDCP500 (500 images from HazeFlow's GitHub) but the clean images (rgb_500) are only on Baidu Disk (inaccessible). Depth maps (da_depth_500) are on Google Drive: https://drive.google.com/drive/folders/1mH36eROxST_-MR9drCWUrJrfylhte2GI — download later if needed for HazeFlow comparison.

### Current Raw Download Locations

```
~/Desktop/leftImg8bit_trainvaltest/
└── leftImg8bit/
    ├── test/
    │   ├── berlin/
    │   ├── bielefeld/
    │   ├── bonn/
    │   ├── leverkusen/
    │   ├── mainz/
    │   └── munich/
    ├── train/
    │   ├── aachen/
    │   ├── bochum/
    │   ├── bremen/
    │   ├── cologne/
    │   ├── darmstadt/
    │   ├── dusseldorf/
    │   ├── erfurt/
    │   ├── hamburg/
    │   ├── hanover/
    │   ├── jena/
    │   ├── krefeld/
    │   ├── monchengladbach/
    │   ├── strasbourg/
    │   ├── stuttgart/
    │   ├── tubingen/
    │   ├── ulm/
    │   ├── weimar/
    │   └── zurich/
    └── val/
        ├── frankfurt/
        ├── lindau/
        └── munster/

~/Desktop/gtFine_trainvaltest (1)/
└── gtFine/
    ├── test/{same cities as above}/
    ├── train/{same cities as above}/
    └── val/{same cities as above}/
```

### Target Project Directory

```
~/Desktop/SceneHaze/
├── SCENEHAZE_PROJECT_CONTEXT.md    ← this file
├── data/
│   ├── clean/
│   │   └── cityscapes/             ← move/symlink leftImg8bit contents here
│   │       ├── train/{city_name}/*.png
│   │       ├── val/{city_name}/*.png
│   │       └── test/{city_name}/*.png
│   ├── depth/
│   │   └── cityscapes/             ← generate with Depth Anything V2
│   │       ├── train/{city_name}/*.png
│   │       ├── val/{city_name}/*.png
│   │       └── test/{city_name}/*.png
│   ├── segmentation/
│   │   └── cityscapes/             ← from gtFine *_labelIds.png files
│   │       ├── train/{city_name}/*_labelIds.png
│   │       ├── val/{city_name}/*_labelIds.png
│   │       └── test/{city_name}/*_labelIds.png
│   ├── generated/
│   │   ├── constant_beta/          ← baseline 1 output
│   │   ├── mcbm/                   ← baseline 2 output (HazeFlow)
│   │   └── scene_grounded/         ← our method output
│   └── evaluation/
│       ├── nh_haze/                ← download from data.vision.ee.ethz.ch
│       └── dnh_haze/               ← NTIRE 2024 challenge data
├── scripts/
│   ├── setup_data.py               ← organize Cityscapes into project structure
│   ├── extract_depth.py            ← run Depth Anything V2
│   ├── generate_constant.py        ← constant-β fog
│   ├── generate_mcbm.py            ← HazeFlow's MCBM baseline
│   └── generate_grounded.py        ← our scene-grounded fog
└── results/
    ├── comparisons/
    └── metrics/
```

### Cityscapes Label IDs (for segmentation-driven β)

Cityscapes `_labelIds.png` uses these pixel values:

| ID | Class | Fog Category | β Modifier |
|----|-------|-------------|-----------|
| 0 | unlabeled | Everything else | ×1.0 |
| 7 | road | Ground/road | ×1.3 |
| 8 | sidewalk | Ground/road | ×1.3 |
| 9 | parking | Ground/road | ×1.3 |
| 10 | rail track | Ground/road | ×1.3 |
| 11 | building | Structures | ×1.0 |
| 12 | wall | Structures | ×1.0 |
| 13 | fence | Structures | ×1.0 |
| 14 | guard rail | Structures | ×1.0 |
| 15 | bridge | Structures | ×1.0 |
| 16 | tunnel | Structures | ×1.0 |
| 17 | pole | Everything else | ×1.0 |
| 18 | polegroup | Everything else | ×1.0 |
| 19 | traffic light | Everything else | ×1.0 |
| 20 | traffic sign | Everything else | ×1.0 |
| 21 | vegetation | Low vegetation | ×1.1 |
| 22 | terrain | Low vegetation | ×1.1 |
| 23 | sky | Sky | ×0.3 |
| 24 | person | Everything else | ×1.0 |
| 25 | rider | Everything else | ×1.0 |
| 26 | car | Everything else | ×1.0 |
| 27 | truck | Everything else | ×1.0 |
| 28 | bus | Everything else | ×1.0 |
| 29 | caravan | Everything else | ×1.0 |
| 30 | trailer | Everything else | ×1.0 |
| 31 | train | Everything else | ×1.0 |
| 32 | motorcycle | Everything else | ×1.0 |
| 33 | bicycle | Everything else | ×1.0 |
| -1 or 255 | license plate / ignore | Everything else | ×1.0 |

---

## Implementation Steps

### Step 1: Organize Directory ← DO THIS FIRST
Move/symlink downloaded Cityscapes data into the SceneHaze project structure:
- `leftImg8bit/{train,val,test}/*` → `SceneHaze/data/clean/cityscapes/{train,val,test}/*`
- `gtFine/{train,val,test}/*` → `SceneHaze/data/segmentation/cityscapes/{train,val,test}/*`

### Step 2: Generate Depth Maps
Run Depth Anything V2 on Cityscapes clean images. Start with 3-5 images only.

```bash
pip install transformers torch torchvision
```

### Step 3: Implement Fog Generation (Piece by Piece)

**Piece 1 — Constant β fog on ONE image:**
```python
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

clean = np.array(Image.open("path/to/clean.png")).astype(float) / 255.0
depth = np.array(Image.open("path/to/depth.png")).astype(float)
depth = (depth - depth.min()) / (depth.max() - depth.min())

beta = 1.0
A = np.array([0.85, 0.85, 0.85])
t = np.exp(-beta * depth)
foggy = clean * t[:,:,np.newaxis] + A * (1 - t[:,:,np.newaxis])

Image.fromarray((foggy * 255).clip(0,255).astype(np.uint8)).save("test_constant.png")
```

**Piece 2 — Scene-grounded β fog on SAME image:**
```python
seg = np.array(Image.open("path/to/labelIds.png"))

beta_base = 1.0
beta_map = np.ones_like(depth) * beta_base

# Apply scene-grounded modifiers using Cityscapes label IDs
beta_map[seg == 23] *= 0.3                           # sky
beta_map[np.isin(seg, [7, 8, 9, 10])] *= 1.3         # road, sidewalk, parking, rail
beta_map[np.isin(seg, [21, 22])] *= 1.1               # vegetation, terrain

# Smooth transitions
beta_map = gaussian_filter(beta_map, sigma=15)

t = np.exp(-beta_map * depth)
foggy = clean * t[:,:,np.newaxis] + A * (1 - t[:,:,np.newaxis])

Image.fromarray((foggy * 255).clip(0,255).astype(np.uint8)).save("test_grounded.png")
```

**Piece 3 — MCBM baseline:**
```bash
git clone https://github.com/cloor/HazeFlow.git
cd HazeFlow && pip install -r requirements.txt
python haze_generation/brownian_motion_generation.py
```

**Piece 4 — Visual comparison:** Put constant vs grounded vs MCBM vs real NH-HAZE side by side.

**Piece 5 — Scale to full dataset** with severity variation (β_base sampled from [0.4, 2.0]).

### Step 4: Benchmark Restoration Models (Later — Needs Kaya HPC)
Run 5-6 SOTA models with original weights, then fine-tune best on each fog type.

Models: OneRestore, WeatherDiff, MWFormer, PromptIR, NAFNet, T3-DiffWeather

### Step 5: Evaluate on Real Benchmarks
Test on: DNH-HAZE (2024), NH-HAZE (2020), O-HAZE, SOTS
Metrics: PSNR, SSIM, LPIPS, optionally FID/FADE

---

## Key Papers

| Paper | Role |
|-------|------|
| **RESIDE** (Li 2019) | Baseline 1 — constant-β dataset everyone trains on |
| **HazeFlow** (Shin, ICCV 2025) | Baseline 2 — MCBM. Code: github.com/cloor/HazeFlow |
| **NH-HAZE** (Ancuti 2020) | Real benchmark — proves the problem exists |
| **DNH-HAZE** (NTIRE 2024) | Latest real benchmark — 374 participants |
| **GenDeg** (Rajagopalan 2024) | Structural template |
| **HazeGAN** (Chen 2022) | Optional baseline 3 — depth-aware, not semantics-aware |
| **RTE** (Beregi-Kovacs 2025) | Related — improved equation, not inputs |
| **IntrinsicWeather** (CVPR 2026) | Related — material-aware editing, not for restoration training |
| **HazeSpace2M** (Islam, MM 2024) | Supervisor's paper |
| **LoLI-Street** (Anwar) | Supervisor's paper |

---

## Dependencies

```bash
# Core (fog generation — no GPU needed)
pip install numpy pillow matplotlib scipy

# Depth extraction (light GPU or Colab)
pip install transformers torch torchvision

# MCBM baseline
git clone https://github.com/cloor/HazeFlow.git
```

---

## What To Do Right Now

1. Organize directory: move Cityscapes data into SceneHaze/data/ structure
2. Run Depth Anything V2 on 3-5 Cityscapes images to get depth maps
3. Implement Piece 1: constant-β ASM fog on ONE image (sanity check)
4. Implement Piece 2: scene-grounded β fog on SAME image using gtFine labelIds
5. Visual comparison of both outputs
6. Iterate on multipliers and smoothing sigma until it looks right
