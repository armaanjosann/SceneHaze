"""
Shared helpers for the rain pipeline (constant-rate and scene-grounded).
Mirrors fog_utils.py's role: one place for the math/compositing primitives
so both rain generators use the exact same equations, and so the manifest's
per-arm stats (compute_rain_stats.py, not yet written) can reuse the same
mask-building functions rather than recomputing them differently.

Five components, composited in order per image (see generate_rain_grounded.py
for the full call sequence; generate_rain_constant.py skips components 1-2):
  1. Wet surface darkening   (grounded only)
  2. Wet surface reflections (grounded only)
  3. Global atmosphere shift (gamma + blue channel boost)
  4. Veiling                 (reuses fog's build_beta_map machinery)
  5. Streaks                 (composited last)

Design note on seeding (locked after design review): TWO independent seeds
drive per-image randomness, deliberately NOT sharing one RNG stream, so a
future change to one component's random-draw sequence (e.g. adding a new
jitter term to streaks) can't silently reshuffle the other component's
already-generated values:
  - "atmosphere" seed: drives sample_atmosphere_shift()'s gamma/blue_boost
    draws. Derived directly from (image, global_seed) via
    derive_atmosphere_seed() -- NEVER stored in the manifest. gamma/
    blue_boost are themselves also never stored (derived at generation
    time from rain_rate + this seed) -- matches the project's established
    pattern of storing seeds, not derived quantities (cf. fog's shuffled
    arm, which stores shuffle_seed but never the resulting permutation).
  - "streak" seed: drives build_streak_layer()'s per-streak draws (angle/
    length/thickness/position). Derived the same way via
    derive_streak_seed(), and IS stored in the manifest as
    rain.streak_seed (the field already exists in the schema) -- because
    unlike the atmosphere draws, downstream stats/debugging code needs to
    be able to reproduce exactly which streaks were drawn without also
    reconstructing gamma/blue_boost, so it gets its own persisted seed
    rather than being re-derived from something else stored.
  Both derivations depend only on (image, global_seed) -- neither is
  derived FROM the other -- so populate_rain_params.py only ever needs to
  write ONE new seed field (streak_seed); the atmosphere seed needs no
  manifest storage at all, it's recomputed from `image` (already present
  on every entry) plus the hardcoded RAIN_GLOBAL_SEED below.

Attribution: the veiling component's functional form (beta_veiling scales
with rain_rate ** 0.67) follows the rain-veiling extinction model used by
Tremblay et al., "Rain Rendering for Evaluating and Improving Robustness
to Bad Weather" (IJCV 2020), who in turn cite Weber et al. 2015 ("A
multiscale model for rain rendering in real-time") for it. Tremblay's own
calibrated constant is 0.312, fit against METRIC depth (their formula
divides depth by 1000 to convert mm->km) -- not directly transferable to
our normalized [0,1] Depth-Anything pseudo-depth, hence RAIN_VEILING_C
below is a placeholder pending our own FID-based calibration (see
generate_grounded.py's fid_calibration.py precedent for fog). The general
idea of rain-rate-dependent streak density/thickness and depth-modulated
streak alpha is consistent with the physical streak model described by
Garg & Nayar, "Photorealistic Rendering of Rain Streaks" (TOG 2006, cited
by Tremblay for their streak texture database) -- but our streak renderer
below is our own simple alpha-blended line-segment implementation, not
adapted from either paper's code (see the rain-rendering/DAF-Net audit:
neither codebase's streak or wet-surface logic was reused here).

Nothing in this module requires 3D geometry, camera intrinsics, or a
particle simulator -- by design, matching our Cityscapes-only, depth-
normalized, segmentation-driven approach.
"""

import hashlib
import math
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, maximum_filter

from generate_grounded import build_beta_map, DEFAULT_SIGMA

# --- global seed + per-image seed derivation --------------------------------

RAIN_GLOBAL_SEED = 42  # same value as fog's GLOBAL_SEED, but its own named
# constant -- rain stays independently reseedable from fog even though they
# happen to start at the same number today.


def derive_atmosphere_seed(image: str, global_seed: int = RAIN_GLOBAL_SEED) -> int:
    """Deterministic per-image seed for sample_atmosphere_shift(). Mirrors
    generate_shuffled.py's derive_shuffle_seed() (sha256, not Python's
    built-in hash() -- see that module for why). The "_atmosphere_" salt in
    the hashed key is what keeps this independent from derive_streak_seed()
    below despite both starting from the same (image, global_seed) pair."""
    key = f"{image}_atmosphere_{global_seed}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big") % (2 ** 31)


def derive_streak_seed(image: str, global_seed: int = RAIN_GLOBAL_SEED) -> int:
    """Deterministic per-image seed for build_streak_layer(). IS persisted
    in the manifest (rain.streak_seed) by populate_rain_params.py, unlike
    the atmosphere seed -- see module docstring for why."""
    key = f"{image}_streak_{global_seed}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big") % (2 ** 31)


# --- category label IDs (Cityscapes labelIds) --------------------------------
# Finer-grained than fog's GROUND_IDS/VEGETATION_IDS (generate_grounded.py),
# which lump road+sidewalk+parking+rail into one modifier and
# vegetation+terrain into another. Components 1-2 below need each category
# weighted individually (road != sidewalk != parking, vegetation != terrain),
# so these are separate constants, not a reuse of fog's coarser groups.
# Component 4 (veiling) reuses fog's coarse groups directly, unchanged.

ROAD_ID = 7
SIDEWALK_ID = 8
PARKING_ID = 9
RAIL_ID = 10
VEGETATION_ID = 21
TERRAIN_ID = 22

# --- component 1: wet surface darkening --------------------------------------

WET_MASK_WEIGHTS = {
    ROAD_ID: 0.15,
    SIDEWALK_ID: 0.15,
    TERRAIN_ID: 0.10,
    VEGETATION_ID: 0.0,  # changed from 0.05 (design review): leaves/grass
    # don't wet-darken the way paved surfaces do -- only road/sidewalk/
    # terrain are genuine wet-darkening surfaces. Left in the dict
    # (rather than omitted) for the same documentation reason PARKING_ID/
    # RAIL_ID stay listed at 0.0 -- explicit is clearer than implicit here.
    PARKING_ID: 0.0,
    RAIL_ID: 0.0,
}
WET_MASK_SIGMA = 10.0
RAIN_SEVERITY_CAP = 100.0  # denominator for min(rain_rate/100, 1.0), shared
# by the wet mask, reflection mask, and streak-thickness scaling below.


def build_wet_mask(seg: np.ndarray, rain_rate: float, sigma: float = WET_MASK_SIGMA) -> np.ndarray:
    """Per-pixel wet-darkening strength in [0, ~0.15]: category weight,
    Gaussian-smoothed so category boundaries don't create harsh edges (same
    rationale as fog's beta_map smoothing), scaled by rain severity.
    Returned as its own array (not folded into apply_wet_darkening) so it
    can double as the aux C1 source and a future wet_ref_corr/
    wet_depth_corr stat input."""
    mask = np.zeros(seg.shape, dtype=np.float32)
    for label_id, weight in WET_MASK_WEIGHTS.items():
        mask[seg == label_id] = weight
    mask = gaussian_filter(mask, sigma=sigma)
    mask *= min(rain_rate / RAIN_SEVERITY_CAP, 1.0)
    return mask


def apply_wet_darkening(clean: np.ndarray, wet_mask: np.ndarray) -> np.ndarray:
    """J_wet = J * (1 - wet_mask). Stays in [0,1] automatically since
    wet_mask in [0,1] and clean in [0,1] -- no clip needed."""
    return clean * (1.0 - wet_mask[:, :, np.newaxis])


WET_SHINE_KERNEL_SIZE = 15  # pixels, maximum_filter neighborhood used to
# broaden the reflection-sourced brightness into wider highlight patches.
WET_SHINE_INTENSITY_SCALE = 0.15  # peak shine_intensity at full severity
# scaling (same min(rain_rate/100,1.0) cap wet_mask/reflection_mask use).
WET_SHINE_SKY_FLOOR_SCALE = 0.6  # floor brightness (as a fraction of
# atmospheric_light) used wherever the reflection itself is dim -- e.g. a
# column reflecting a dark building facade rather than open sky. Without
# this floor, shine would vanish exactly where a wet road often still
# shows real-world highlight (ambient sky-glow, not a direct mirror hit).


def normalize_wet_gate(wet_mask: np.ndarray) -> np.ndarray:
    """Normalizes wet_mask to its own [0,1] range for use as a spatial
    GATE in shine/saturation -- NOT for darkening, which uses raw
    wet_mask (correctly calibrated at ~15% peak, confirmed by design
    review). wet_mask's raw peak (~0.15) was sized for darkening;
    reusing it directly as an intensity multiplier for shine/saturation
    silently capped both at ~15% of their stated value (see v8_shine/
    v9_sat's near-invisible results and diagnostics -- this was the
    root cause). Normalizing lets those two effects reach their full
    stated intensity on the wettest pixels while darkening itself is
    untouched."""
    peak = float(wet_mask.max())
    if peak < 1e-8:
        return np.zeros_like(wet_mask)
    return np.clip(wet_mask / peak, 0.0, 1.0)


def apply_wet_shine(J_wet: np.ndarray, wet_mask: np.ndarray, rain_rate: float, reflection: np.ndarray, atmospheric_light: np.ndarray, kernel_size: int = WET_SHINE_KERNEL_SIZE) -> np.ndarray:
    """Adds specular highlights to wet surfaces so they read as wet-and-
    shiny rather than just uniformly darker -- component 1 alone
    (darkening only) can look like a dirty/shadowed road rather than a
    rain-wet one.

    SECOND PASS (first pass max-pooled J_wet's own already-dark pixels --
    physically backwards, and produced a near-invisible max shine of
    0.0125; see v8_shine's diagnostics). Brightness now comes from
    `reflection` -- the per-column contact-row mirror already computed by
    component 2 (build_reflection), broadened via maximum_filter into
    wider highlight patches. This is physically motivated: a wet surface
    shines because it mirrors the sky/scene above it, not because it
    borrows brightness from its own dark pixels. Wherever the mirrored
    content is itself dim (e.g. reflecting a dark building facade), a
    floor of `atmospheric_light * WET_SHINE_SKY_FLOOR_SCALE` is blended
    in via elementwise max, so shine doesn't vanish there.

    Gated by normalize_wet_gate(wet_mask) (see that function for why
    raw wet_mask can't be used directly here), scaled by
    shine_intensity = WET_SHINE_INTENSITY_SCALE * severity_scale, same
    min(rain_rate/100,1.0) severity cap as every other quantity in this
    module.

    Must run AFTER `reflection` is computed (build_reflection) and AFTER
    wet darkening, BEFORE apply_reflections' own compositing step and
    before atmosphere shift -- see generate_rain_grounded.py for the
    call order.
    """
    shine_intensity = WET_SHINE_INTENSITY_SCALE * min(rain_rate / RAIN_SEVERITY_CAP, 1.0)
    gate = normalize_wet_gate(wet_mask)

    reflection_brightness = np.stack(
        [maximum_filter(reflection[:, :, c], size=kernel_size) for c in range(3)], axis=-1
    )
    sky_floor = np.asarray(atmospheric_light, dtype=np.float32).reshape(1, 1, 3) * WET_SHINE_SKY_FLOOR_SCALE
    brightness_source = np.maximum(reflection_brightness, sky_floor)

    shine = gate[:, :, np.newaxis] * shine_intensity * brightness_source
    return np.clip(J_wet + shine, 0.0, 1.0)


WET_SATURATION_BOOST = 0.3  # peak fractional saturation increase at
# full wet_mask + full rain_rate severity scaling (30% at full strength).
# First empirical pass, per spec -- dial back to 0.15-0.20 if it over-
# saturates.


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB->HSV for float arrays in [0,1], shape (...,3). Pure
    numpy (no matplotlib/cv2 dependency -- matplotlib.colors.rgb_to_hsv
    would work too and is available in this env, but everything else in
    this module only depends on numpy/scipy/PIL, so keeping it that way
    rather than adding a new import for two small functions). Standard
    formula, equivalent to colorsys.rgb_to_hsv applied per-pixel but
    vectorized -- cross-checked against colorsys on random inputs before
    use (see commit history / validation script)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 1e-8, delta / np.maximum(maxc, 1e-8), 0.0)

    delta_safe = np.where(delta > 1e-8, delta, 1.0)  # avoid div-by-zero; unused where delta==0 (s==0, h forced to 0 below)
    rc = (maxc - r) / delta_safe
    gc = (maxc - g) / delta_safe
    bc = (maxc - b) / delta_safe

    is_r = maxc == r
    is_g = (maxc == g) & ~is_r
    is_b = (maxc == b) & ~is_r & ~is_g

    h = np.zeros_like(maxc)
    h = np.where(is_r, bc - gc, h)
    h = np.where(is_g, 2.0 + rc - bc, h)
    h = np.where(is_b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta > 1e-8, h, 0.0)

    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_hsv, same conventions."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i.astype(np.int64) % 6

    r = np.select([i_mod == k for k in range(6)], [v, q, p, p, t, v])
    g = np.select([i_mod == k for k in range(6)], [t, v, v, q, p, p])
    b = np.select([i_mod == k for k in range(6)], [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=-1)


def apply_wet_saturation(J: np.ndarray, wet_mask: np.ndarray, rain_rate: float, saturation_boost: float = WET_SATURATION_BOOST) -> np.ndarray:
    """Color intensification on wet surfaces -- wet asphalt looks richer
    in color, not just darker, which darkening (component 1) and shine
    don't capture on their own. Converts to HSV, scales the S channel by
    (1 + saturation_boost * gate * severity_scale), converts back.
    SECOND PASS: gate is normalize_wet_gate(wet_mask), not raw wet_mask --
    the raw version capped this effect at ~15% of its stated value (see
    v9_sat's 4.3% actual vs. 30% nominal; normalize_wet_gate's docstring
    has the full root-cause explanation). Severity scaling reuses the
    same min(rain_rate/100,1.0) cap as every other quantity in this
    module. Own function (not folded into apply_wet_shine) to match this
    module's one-function-per-physical-effect convention."""
    hsv = rgb_to_hsv(np.clip(J, 0.0, 1.0))
    gate = normalize_wet_gate(wet_mask)
    boost = saturation_boost * gate * min(rain_rate / RAIN_SEVERITY_CAP, 1.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + boost), 0.0, 1.0)
    return np.clip(hsv_to_rgb(hsv), 0.0, 1.0)


# --- component 2: wet surface reflections ------------------------------------

REFLECTION_MASK_WEIGHTS = {
    ROAD_ID: 0.20,  # bumped from 0.08 (v10 wet-surface pass): the direct
    # reflection composite (apply_reflections, separate from shine's now-
    # reflection-sourced brightness) was measured too subtle to read as
    # visibly wet (mean contribution 0.022 at road in the v3-v5 pass).
    # This is the one wet-surface mechanism that's both geometrically
    # validated (post blur-order fix) and physically correct, so pushed
    # up here rather than adding another parallel weak effect.
    SIDEWALK_ID: 0.12,  # bumped from 0.05, same ratio to ROAD_ID preserved (~1.7x)
    # everything else defaults to 0.0
}
REFLECTION_BLUR_SIGMA = 30.0
REFLECTION_MASK_SIGMA = 15.0


def compute_contact_rows(seg: np.ndarray) -> np.ndarray:
    """Per-column: the row index of the topmost (smallest y) road-or-
    sidewalk pixel, or -1 if the column has none. Vectorized via argmax
    over a boolean column mask (not a Python loop over ~2048 columns,
    which would be the main perf risk here) -- argmax returns the index
    of the first True per column; where no True exists at all it defaults
    to 0, which we explicitly override to -1 via `has_ground`."""
    ground_mask = np.isin(seg, [ROAD_ID, SIDEWALK_ID])
    has_ground = ground_mask.any(axis=0)
    raw_argmax = np.argmax(ground_mask, axis=0)
    return np.where(has_ground, raw_argmax, -1)


def build_reflection(clean: np.ndarray, contact_rows: np.ndarray, sigma: float = REFLECTION_BLUR_SIGMA) -> np.ndarray:
    """Per-column contact-row mirror (v1.5): for each column with a valid
    contact row y_c, rows y >= y_c are filled by mirroring the image about
    y_c (reflected[y,x] = clean[clip(2*y_c - y, 0, H-1), x]) -- reflecting
    the scene above the contact row down onto the ground plane below it.
    Columns with no road/sidewalk pixel at all, and rows above their
    column's own contact row, pass through unchanged (= clean). This is a
    deliberate design choice (confirmed): the reflection MASK (see
    build_reflection_mask) is zero in exactly those same no-ground
    columns, so this pass-through data never actually shows through --
    it's a don't-care value, not a visible artifact.

    Vectorized via a per-pixel gather (fancy indexing on two (H,W) integer
    arrays), not a Python loop over columns.

    BUG FIX (found during diagnostic review, see commit history): the
    sigma=30 blur is applied to the MIRRORED content BEFORE compositing
    with the pass-through region, not after. The original order blurred
    the entire combined array -- mirror AND pass-through together -- which
    at sigma=30 corrupted pixels that were supposed to stay pixel-
    identical to `clean` (verified: row 0, guaranteed pass-through since
    it's always above every column's contact row, differed from clean by
    a mean of 0.126 when it should have been exactly 0.0). Blurring first
    keeps pass-through pixels exactly sharp/unchanged and only softens the
    actual reflected content, as originally intended. One side effect of
    the fix: the boundary between sharp pass-through and blurred
    reflection is now a real (if soft-edged-on-one-side) discontinuity,
    not smoothed away by blur bleed across it -- worth checking visually,
    not assumed to be fine.
    """
    h, w = clean.shape[:2]
    rows = np.arange(h).reshape(h, 1)
    cols = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))

    valid_cols = contact_rows >= 0
    contact_rows_safe = np.where(valid_cols, contact_rows, 0).reshape(1, w)  # dummy 0 for invalid cols, masked out below

    src_rows = np.clip(2 * contact_rows_safe - rows, 0, h - 1)
    mirrored = clean[src_rows, cols]
    mirrored = np.stack([gaussian_filter(mirrored[:, :, c], sigma=sigma) for c in range(3)], axis=-1)  # blur BEFORE compositing

    apply_region = (rows >= contact_rows_safe) & valid_cols.reshape(1, w)

    # Known limitation, accepted as-is after visual review: this mirror is
    # purely row-based and has no object awareness, so a car (or anything
    # else) sitting on the road gets mirrored into the reflection just like
    # the road surface itself -- this can read as a faint
    # shadow/reflection artifact directly under vehicles. The blur here,
    # the rain_rate severity scaling, and the reflection mask's own low
    # peak intensity (0.08 for road) all combine to keep this subtle
    # rather than jarring. Revisit only if visual sample review says
    # otherwise -- not treated as a bug to fix blind.
    return np.where(apply_region[:, :, np.newaxis], mirrored, clean)


def build_reflection_mask(seg: np.ndarray, rain_rate: float, sigma: float = REFLECTION_MASK_SIGMA) -> np.ndarray:
    """Same shape as build_wet_mask but with REFLECTION_MASK_WEIGHTS and
    its own (larger) smoothing sigma."""
    mask = np.zeros(seg.shape, dtype=np.float32)
    for label_id, weight in REFLECTION_MASK_WEIGHTS.items():
        mask[seg == label_id] = weight
    mask = gaussian_filter(mask, sigma=sigma)
    mask *= min(rain_rate / RAIN_SEVERITY_CAP, 1.0)
    return mask


def apply_reflections(J_wet: np.ndarray, reflection: np.ndarray, reflection_mask: np.ndarray) -> np.ndarray:
    """J_wet_reflect = J_wet + reflection * reflection_mask. No clip here
    (could exceed 1.0) -- this module clips exactly once, at the very end
    of the whole pipeline (apply_streaks), matching fog's single end-of-
    apply_asm clip; clipping mid-pipeline would double-attenuate highlights
    an earlier step already pushed out of range."""
    return J_wet + reflection * reflection_mask[:, :, np.newaxis]


# --- component 3: global atmosphere shift ------------------------------------

GAMMA_BASE = 0.95
GAMMA_RAIN_SCALE = 0.15
GAMMA_RAIN_RATE_CAP = 150.0  # gamma's own denominator -- deliberately
# different from RAIN_SEVERITY_CAP (100.0) used elsewhere, per spec.
GAMMA_JITTER = 0.05  # +/- uniform range
BLUE_BOOST_RANGE = (1.02, 1.08)


def sample_atmosphere_shift(image: str, rain_rate: float, global_seed: int = RAIN_GLOBAL_SEED) -> tuple:
    """Returns (gamma, blue_boost), deterministic per (image, rain_rate,
    global_seed). Draw order fixed: gamma's jitter first, then blue_boost
    -- from a single random.Random(atmosphere_seed) instance, so the
    sequence is reproducible. Never stored in the manifest -- see module
    docstring.

    IMPORTANT, DO NOT REORDER: reproducibility depends on gamma's jitter
    being drawn from `rng` before blue_boost. Swapping the two draws (or
    inserting a new draw between them) silently changes every
    already-generated image's gamma/blue_boost pair -- and since neither
    is stored in the manifest, there would be no way to detect the drift
    after the fact except by re-deriving and diffing against
    previously-saved RGB output.
    """
    seed = derive_atmosphere_seed(image, global_seed)
    rng = random.Random(seed)
    gamma = GAMMA_BASE - GAMMA_RAIN_SCALE * min(rain_rate / GAMMA_RAIN_RATE_CAP, 1.0) + rng.uniform(-GAMMA_JITTER, GAMMA_JITTER)
    blue_boost = rng.uniform(*BLUE_BOOST_RANGE)
    return gamma, blue_boost


def apply_atmosphere_shift(J: np.ndarray, A: np.ndarray, gamma: float, blue_boost: float) -> tuple:
    """Applies gamma correction (all 3 channels) then a blue-channel boost
    (channel 2 only) to J and to A together, then clips both to [0,1] --
    confirmed order (design review). `np.clip(J, 0, None)` floors J at 0
    before the fractional-power gamma (x**gamma is NaN for x<0 with
    non-integer gamma); J should never actually be negative at this point
    but the floor is cheap insurance."""
    J_shifted = np.clip(J, 0.0, None) ** gamma
    J_shifted[:, :, 2] *= blue_boost
    J_shifted = np.clip(J_shifted, 0.0, 1.0)

    A_shifted = np.clip(np.asarray(A, dtype=np.float32), 0.0, None) ** gamma
    A_shifted = A_shifted.copy()
    A_shifted[2] *= blue_boost
    A_shifted = np.clip(A_shifted, 0.0, 1.0)

    return J_shifted, A_shifted


# --- component 4: veiling (reuses fog's build_beta_map) ----------------------

RAIN_VEILING_C = 0.03  # rough empirical calibration (2026-09-02, second
# pass): c=1.0 gave a ~0.0065 mean transmission (scene fully obscured);
# c=0.1 improved it to ~0.18 mean transmission (scene visible but still
# heavier haze than intended) -- both rejected by visual + numeric review.
# 0.03 targets beta~0.73 (peak ~1.0 after modifiers) and transmission in
# the ~0.4-0.6 range at rain_rate~120: scene should retain color/contrast
# with mild haze at distance only, not lose structure. NOT yet a proper
# calibration. BDD100K FID calibration (mirroring fog's
# fid_calibration.py) is still future work and may move this further;
# iterate empirically (see the visual sample review) before that lands.


def compute_veiling_beta_map(seg: np.ndarray, depth: np.ndarray, rain_rate: float, c: float = RAIN_VEILING_C, sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    """Grounded arm: scene-grounded veiling, reusing fog's build_beta_map
    verbatim (same sky/road/veg modifiers, same depth-scaling, same
    Gaussian smoothing) with beta_base = c * rain_rate ** 0.67 in place of
    fog's uniform-random beta_base."""
    beta_scale = c * (rain_rate ** 0.67)
    return build_beta_map(seg, depth, beta_scale, sigma)


def compute_veiling_beta_uniform(rain_rate: float, shape: tuple, c: float = RAIN_VEILING_C) -> np.ndarray:
    """Constant arm: uniform (non-seg-modified) veiling -- the same
    rain_rate-derived scalar broadcast across every pixel, mirroring fog's
    constant arm using a scalar beta instead of build_beta_map."""
    beta_scale = c * (rain_rate ** 0.67)
    return np.full(shape, beta_scale, dtype=np.float32)


def apply_veiling(J_shifted: np.ndarray, depth: np.ndarray, beta_map: np.ndarray, A_shifted: np.ndarray) -> np.ndarray:
    """I_veiled = J_shifted*t + A_shifted*(1-t), t = exp(-beta_map*depth) --
    the same ASM formula as fog_utils.apply_asm, but A is the per-image
    A_shifted (post atmosphere-shift) rather than fog's fixed constant.
    No clip needed: t in [0,1], and J_shifted/A_shifted are already
    clipped to [0,1] by apply_atmosphere_shift, so this convex combination
    stays in [0,1] automatically."""
    t = np.exp(-beta_map * depth)
    return J_shifted * t[:, :, np.newaxis] + A_shifted * (1 - t[:, :, np.newaxis])


# --- component 5: streaks (composited last) ----------------------------------

STREAKS_PER_MM = 8  # n_streaks = int(rain_rate * STREAKS_PER_MM), capped by
# MAX_STREAK_COUNT below.
MAX_STREAK_COUNT = 1200  # physically-motivated ceiling. A no-op given
# rain_rate's populated range ([10,150] -> at most 150*8=1200 exactly) --
# kept as an explicit defensive cap so a CLI override or future range
# change can't silently blow past a sane per-image streak count.
MIN_STREAK_LENGTH_FOR_DRAW = 2.0  # pixels. Rendering-artifact guard, not
# physics: skips the draw call (not the RNG draws -- see below) for a
# streak whose sampled length rounds to something too small to render
# cleanly. Currently unreachable given STREAK_LENGTH_SHORT_RANGE's floor
# of 8px, but kept as a guard against a future range change producing
# degenerate near-zero-length streaks that render as stray dots.
STREAK_BASE_ANGLE_RANGE = (75.0, 105.0)  # degrees from horizontal; 90 deg =
# straight down, so this range is a near-vertical streak with up to +/-15
# deg of wind tilt, drawn once per image as the shared base angle.
# UNCHANGED across both streak-rendering revisions, per explicit instruction.
STREAK_ANGLE_JITTER = 3.0  # unchanged, ditto.
STREAK_LENGTH_SHORT_RANGE = (8.0, 20.0)  # pixels (2nd pass; was (10,15))
STREAK_LENGTH_LONG_RANGE = (25.0, 55.0)  # pixels (2nd pass; was (30,50))
STREAK_SHORT_FRACTION = 0.6  # 60% short / 40% long (2nd pass; was 50/50) --
# still a genuine bimodal mixture (12-25px gap between tiers), not a
# single widened uniform range (the very original STREAK_LENGTH_RANGE=
# (20,60)).
STREAK_ALPHA_RANGE = (0.3, 0.55)  # each streak independently samples its
# own peak opacity (before depth fade) from this range -- replaces the
# first pass's single shared STREAK_BASE_ALPHA=0.45 constant. Real rain
# streak opacity varies streak-to-streak (droplet size/lighting); a
# single shared constant couldn't capture that, and was still part of
# the "painted on, uniform brightness" look flagged after that first pass.
STREAK_BRIGHTNESS_RANGE = (0.75, 1.0)  # NEW (2nd pass): per-streak color
# intensity multiplying STREAK_COLOR (1.0 = pure white, lower = dimmer/
# grayer). Distinct from STREAK_ALPHA_RANGE: alpha is how much of the
# streak's color shows through vs. the scene beneath it; brightness is
# what that color actually is. Both now vary per streak.
STREAK_BLUR_SIGMA = 0.75  # gaussian blur applied to EACH streak's own
# small local patch before compositing (see build_streak_layer) -- TRUE
# per-streak blur now (2nd pass), not the first pass's whole-canvas
# approximation; per-streak local canvases were needed anyway once alpha
# and brightness became per-streak, so per-streak blur came for free.
# Anti-aliases the hard PIL line edges -- the single biggest contributor
# to the "painted on" look.
STREAK_THICKNESS_BASE = 1.0
STREAK_THICKNESS_RAIN_SCALE = 2.0
STREAK_THICKNESS_JITTER = 0.5
STREAK_MAX_THICKNESS = 2.0  # NEW (2nd pass): hard cap. Uncapped, the base
# formula (1.0 + 2.0*min(rain_rate/100,1) + jitter) reaches ~3-3.5px at
# high rain_rate -- individually thick, opaque-looking streaks were part
# of the "painted on" look. More streaks + lower alpha at high rain_rate,
# not individually thicker opaque ones.
STREAK_COLOR = (1.0, 1.0, 1.0)  # base white; STREAK_BRIGHTNESS_RANGE
# scales this down per streak (see above).
STREAK_DEPTH_ATTENUATION_C = 1.0  # bumped from 0.5 (2nd pass, visual +
# numeric review): at 0.5 the fade factor only ranged ~[0.61,1.0] over
# the image's full depth range -- a real, verified effect (confirmed via
# direct near/far pixel comparison in the v3 diagnostic: it was NOT a
# no-op), but too mild to read as "distant streaks visibly fading" at a
# glance. 1.0 roughly doubles the attenuation strength. Independent of
# RAIN_VEILING_C -- see the note this constant used to carry, preserved
# here: they intentionally do NOT share a calibration path.


def build_streak_layer(shape: tuple, depth: np.ndarray, rain_rate: float, streak_seed: int, c: float = STREAK_DEPTH_ATTENUATION_C) -> tuple:
    """Returns (streak_alpha, streak_brightness), both float32 (H,W).
    streak_alpha in [0,1] is how much of each pixel is covered by
    (depth-attenuated) streak; streak_brightness (only meaningful where
    streak_alpha>0) is that pixel's streak color intensity, in
    STREAK_BRIGHTNESS_RANGE -- see apply_streaks for how the two combine.

    n_streaks scales linearly with rain_rate (capped at MAX_STREAK_COUNT).
    Each streak's angle/length-tier/length/thickness/position/alpha/
    brightness is drawn from random.Random(streak_seed) in that fixed
    order, one streak at a time -- ALL SEVEN draws happen unconditionally
    for every streak, even ones too short to draw (see
    MIN_STREAK_LENGTH_FOR_DRAW), so RNG consumption never depends on the
    render outcome. IMPORTANT, DO NOT REORDER: same reproducibility
    rationale as sample_atmosphere_shift's fixed draw order.

    Each streak is rendered on its OWN small local canvas (sized to its
    bounding box, not the full image), blurred individually
    (STREAK_BLUR_SIGMA), scaled by its own sampled alpha, then
    max-composited into the shared accumulators: wherever a streak's
    local alpha exceeds what's already accumulated at a pixel, that
    streak's brightness "wins" at that pixel too. This is a reasonable,
    not physically exact, rule for overlapping streaks of different
    brightness -- overlaps are rare at these streak densities (~6% frame
    coverage even at high rain_rate, per the v3 diagnostic), so a more
    expensive true alpha-compositing stack isn't worth it here.

    Depth attenuation (streak_alpha *= exp(-c*depth)) is applied ONCE at
    the end, to the fully-composited alpha layer -- it's a function of
    the scene's depth map, not of any individual streak's own properties.

    Position is sampled as each streak's CENTER point (not its start),
    uniform across the full image -- unchanged from the first pass; still
    not pinned to start/end/center by the spec, flagging in case it needs
    revisiting later.
    """
    h, w = shape
    n_streaks = min(int(rain_rate * STREAKS_PER_MM), MAX_STREAK_COUNT)

    rng = random.Random(streak_seed)
    base_angle = rng.uniform(*STREAK_BASE_ANGLE_RANGE)

    streak_alpha = np.zeros((h, w), dtype=np.float32)
    streak_brightness = np.ones((h, w), dtype=np.float32)  # meaningful only where streak_alpha>0

    for _ in range(n_streaks):
        angle = base_angle + rng.uniform(-STREAK_ANGLE_JITTER, STREAK_ANGLE_JITTER)
        is_short = rng.random() < STREAK_SHORT_FRACTION
        length = rng.uniform(*STREAK_LENGTH_SHORT_RANGE) if is_short else rng.uniform(*STREAK_LENGTH_LONG_RANGE)
        thickness = min(
            STREAK_MAX_THICKNESS,
            max(
                1.0,
                STREAK_THICKNESS_BASE
                + STREAK_THICKNESS_RAIN_SCALE * min(rain_rate / RAIN_SEVERITY_CAP, 1.0)
                + rng.uniform(-STREAK_THICKNESS_JITTER, STREAK_THICKNESS_JITTER),
            ),
        )
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        alpha = rng.uniform(*STREAK_ALPHA_RANGE)
        brightness = rng.uniform(*STREAK_BRIGHTNESS_RANGE)

        if length < MIN_STREAK_LENGTH_FOR_DRAW:
            continue  # RNG already consumed above -- only the draw is skipped

        rad = math.radians(angle)
        dx, dy = math.cos(rad) * length / 2, math.sin(rad) * length / 2
        x0, y0 = cx - dx, cy - dy
        x1, y1 = cx + dx, cy + dy

        pad = thickness / 2 + 3 * STREAK_BLUR_SIGMA + 1  # margin so the blur isn't clipped at the patch edge
        bx0, bx1 = int(np.floor(min(x0, x1) - pad)), int(np.ceil(max(x0, x1) + pad))
        by0, by1 = int(np.floor(min(y0, y1) - pad)), int(np.ceil(max(y0, y1) + pad))

        cbx0, cby0 = max(bx0, 0), max(by0, 0)
        cbx1, cby1 = min(bx1, w), min(by1, h)
        if cbx1 <= cbx0 or cby1 <= cby0:
            continue  # streak's bounding box falls entirely outside the image

        patch = Image.new("L", (bx1 - bx0, by1 - by0), 0)
        patch_draw = ImageDraw.Draw(patch)
        patch_draw.line(
            [(x0 - bx0, y0 - by0), (x1 - bx0, y1 - by0)],
            fill=255, width=max(1, round(thickness)),
        )
        patch_arr = np.asarray(patch, dtype=np.float32) / 255.0
        patch_arr = gaussian_filter(patch_arr, sigma=STREAK_BLUR_SIGMA)
        patch_arr *= alpha

        sub = patch_arr[cby0 - by0: cby1 - by0, cbx0 - bx0: cbx1 - bx0]
        region_alpha = streak_alpha[cby0:cby1, cbx0:cbx1]
        won = sub > region_alpha
        streak_alpha[cby0:cby1, cbx0:cbx1] = np.maximum(region_alpha, sub)
        streak_brightness[cby0:cby1, cbx0:cbx1] = np.where(
            won, brightness, streak_brightness[cby0:cby1, cbx0:cbx1]
        )

    streak_alpha = streak_alpha * np.exp(-c * depth)
    return streak_alpha, streak_brightness


def apply_streaks(I_veiled: np.ndarray, streak_alpha: np.ndarray, streak_brightness: np.ndarray, color: tuple = STREAK_COLOR) -> np.ndarray:
    """Alpha-blended overlay using each pixel's own streak_brightness to
    scale `color` (per-streak brightness variation -- see
    build_streak_layer), then the pipeline's single final clip.
    Mathematically the convex-combination invariant already holds here too
    (streak_alpha in [0,1], I_veiled in [0,1]) -- the clip is defensive,
    matching fog_utils.apply_asm's always-clip convention rather than
    relying on the invariant."""
    color_arr = np.asarray(color, dtype=np.float32)
    streak_color = color_arr[np.newaxis, np.newaxis, :] * streak_brightness[:, :, np.newaxis]
    composited = I_veiled * (1 - streak_alpha[:, :, np.newaxis]) + streak_color * streak_alpha[:, :, np.newaxis]
    return np.clip(composited, 0.0, 1.0)


# --- aux artifact glue --------------------------------------------------------


def rain_aux_channels(veiling_beta_map: np.ndarray, depth: np.ndarray, wet_mask: np.ndarray = None, reflection_mask: np.ndarray = None) -> tuple:
    """Returns (veiling_density, surface_effect) for fog_utils.save_aux's
    C0/C1 args. wet_mask/reflection_mask are None for the constant arm
    (components 1-2 skipped there), giving a legitimately all-zero C1 --
    same convention as fog's constant arm."""
    veiling_density = 1.0 - np.exp(-veiling_beta_map * depth)
    if wet_mask is None and reflection_mask is None:
        surface_effect = np.zeros(depth.shape, dtype=np.float32)
    else:
        wm = wet_mask if wet_mask is not None else np.zeros(depth.shape, dtype=np.float32)
        rm = reflection_mask if reflection_mask is not None else np.zeros(depth.shape, dtype=np.float32)
        surface_effect = np.maximum(wm, rm)
    return veiling_density, surface_effect
