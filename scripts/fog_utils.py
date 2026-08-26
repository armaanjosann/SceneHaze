"""
Shared helpers for ASM-based fog generation (constant-beta and scene-grounded).
Keeping this in one place so both generators use the exact same math and I/O
conventions — no drift between baseline and method.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_ROOT = PROJECT_ROOT / "data" / "clean" / "cityscapes"
DEPTH_ROOT = PROJECT_ROOT / "data" / "depth" / "cityscapes"
SEG_ROOT = PROJECT_ROOT / "data" / "segmentation" / "cityscapes"

# Near-white atmospheric light, per project context doc
ATMOSPHERIC_LIGHT = np.array([0.85, 0.85, 0.85], dtype=np.float32)


def load_clean(split: str, city: str, image: str) -> np.ndarray:
    path = CLEAN_ROOT / split / city / f"{image}_leftImg8bit.png"
    if not path.exists():
        raise SystemExit(f"Missing clean image: {path}")
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def load_disparity(split: str, city: str, image: str) -> np.ndarray:
    path = DEPTH_ROOT / split / city / f"{image}_depth.npy"
    if not path.exists():
        raise SystemExit(
            f"Missing depth map: {path}\n"
            f"Run: python3 scripts/extract_depth.py --split {split} --city {city}"
        )
    return np.load(path)


def load_seg_labels(split: str, city: str, image: str) -> np.ndarray:
    path = SEG_ROOT / split / city / f"{image}_gtFine_labelIds.png"
    if not path.exists():
        raise SystemExit(f"Missing segmentation labels: {path}")
    return np.array(Image.open(path))


def disparity_to_pseudo_depth(disparity: np.ndarray) -> np.ndarray:
    """Invert+normalize Depth Anything disparity (larger = closer) into a
    [0,1] pseudo-depth (larger = farther), matching what the ASM equation
    expects: attenuation should grow with distance, not proximity."""
    d_norm = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-8)
    return 1.0 - d_norm


def apply_asm(clean: np.ndarray, depth: np.ndarray, beta, A: np.ndarray = ATMOSPHERIC_LIGHT) -> np.ndarray:
    """I(x) = J(x)*t(x) + A*(1-t(x)), t(x) = exp(-beta * d(x)).
    `beta` may be a scalar (constant-beta baseline) or a per-pixel array
    the same shape as `depth` (scene-grounded method)."""
    t = np.exp(-beta * depth)
    foggy = clean * t[:, :, np.newaxis] + A * (1 - t[:, :, np.newaxis])
    return np.clip(foggy, 0, 1)


def save_image(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def apply_turbulence(beta_map: np.ndarray, strength: float = 0.15, scale: float = 20.0, seed: int = None) -> np.ndarray:
    """Layer small-scale organic variation on top of a beta map: multiply by
    smooth noise centered at 1.0 (mean-preserving), so pockets of slightly
    thicker/thinner fog appear within an otherwise-uniform category (e.g.
    within "road") without disturbing the large-scale scene structure
    (sky vs. ground) already encoded in beta_map.

    strength: target std dev of the *final* (post-smoothing) noise field —
              bigger = more contrast between thick/thin pockets.
    scale:    gaussian sigma used to smooth the noise (bigger = larger,
              softer blobs; smaller = finer-grained turbulence).
    seed:     for reproducibility — same seed + strength + scale always
              gives the same noise field.

    Note: Gaussian-smoothing a white noise field crushes its amplitude, more
    so the larger `scale` is (e.g. strength=0.15 at scale=20 survives
    smoothing at only ~1% of its original std). So we smooth first, then
    renormalize to the requested std — this keeps `strength` and `scale`
    acting as independent, literal knobs (contrast vs. blob size) instead of
    strength's effect silently depending on scale.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=beta_map.shape)
    noise = gaussian_filter(noise, sigma=scale)
    noise = noise / (noise.std() + 1e-8) * strength
    return beta_map * (1.0 + noise)


# --- shared comparison-figure helpers -------------------------------------

PANEL_SIZE = (1024, 512)  # downscaled per panel so multi-panel grids stay a reasonable file size
LABEL_HEIGHT = 36


def load_and_resize(path: Path, size=PANEL_SIZE) -> Image.Image:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return Image.open(path).convert("RGB").resize(size, Image.LANCZOS)


def label_panel(img: Image.Image, text: str, label_height: int = LABEL_HEIGHT) -> Image.Image:
    """Add a labeled strip on top of a panel."""
    canvas = Image.new("RGB", (img.width, img.height + label_height), (20, 20, 20))
    canvas.paste(img, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), text, fill=(255, 255, 255))
    return canvas


def make_grid(labeled_panels: list, cols: int) -> Image.Image:
    """Arrange same-sized labeled panels into a grid, row-major."""
    rows = (len(labeled_panels) + cols - 1) // cols
    pw, ph = labeled_panels[0].size
    grid = Image.new("RGB", (pw * cols, ph * rows), (0, 0, 0))
    for i, panel in enumerate(labeled_panels):
        r, c = divmod(i, cols)
        grid.paste(panel, (c * pw, r * ph))
    return grid
