"""
Health check: re-derive every file_paths entry in the manifest from
(split, city, image, weather-specific params) using the SAME canonical
naming convention the generator scripts actually use, and diff against
what's stored under file_paths. Reports mismatches. Modifies nothing.

Why this needs to exist at all: file_paths stores strings that are 100%
derivable from identity + params -- which means the manifest holds two
sources of truth (whatever the generators actually name their output
files, and whatever got frozen into the manifest at generation time). If
the naming convention ever changes without every affected entry being
regenerated, file_paths goes stale silently -- nothing else would notice.
Run this after every generation pass, not just once.

Currently only checks `clean` and `fog` (the only weather type with real
data). `rain`/`snow` file_paths are still null pre-generation -- nothing to
verify yet, so they're skipped rather than reported as mismatches.

Usage (NOT executed as part of writing this file):
  python3 scripts/verify_manifest_paths.py
"""

from pathlib import Path

from manifest_io import read_manifest, DEFAULT_MANIFEST_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_ROOT = PROJECT_ROOT / "data" / "clean" / "cityscapes"
CONSTANT_ROOT = PROJECT_ROOT / "data" / "generated" / "constant_beta"
GROUNDED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded"
SHUFFLED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded_shuffled"


def expected_clean_path(split: str, city: str, image: str) -> str:
    return str((CLEAN_ROOT / split / city / f"{image}_leftImg8bit.png").relative_to(PROJECT_ROOT))


def expected_fog_paths(split: str, city: str, image: str, beta_base):
    if beta_base is None:
        return None
    b = f"{beta_base:.2f}"
    constant_stem = f"{image}_beta{b}_constant"
    grounded_stem = f"{image}_betabase{b}_grounded"
    shuffled_stem = f"{image}_betabase{b}_shuffled"
    return {
        "constant": {
            "rgb": str((CONSTANT_ROOT / split / city / f"{constant_stem}.png").relative_to(PROJECT_ROOT)),
            "aux": str((CONSTANT_ROOT / split / city / f"{constant_stem}_aux.npz").relative_to(PROJECT_ROOT)),
        },
        "grounded": {
            "rgb": str((GROUNDED_ROOT / split / city / f"{grounded_stem}.png").relative_to(PROJECT_ROOT)),
            "aux": str((GROUNDED_ROOT / split / city / f"{grounded_stem}_aux.npz").relative_to(PROJECT_ROOT)),
        },
        "shuffled": {
            "rgb": str((SHUFFLED_ROOT / split / city / f"{shuffled_stem}.png").relative_to(PROJECT_ROOT)),
            "aux": str((SHUFFLED_ROOT / split / city / f"{shuffled_stem}_aux.npz").relative_to(PROJECT_ROOT)),
        },
    }


def main():
    manifest = read_manifest(DEFAULT_MANIFEST_PATH)

    mismatches = []
    checked = 0

    for entry in manifest:
        split, city, image = entry["split"], entry["city"], entry["image"]
        fp = entry.get("file_paths", {})

        checked += 1
        expected_clean = expected_clean_path(split, city, image)
        if fp.get("clean") != expected_clean:
            mismatches.append((image, "clean", fp.get("clean"), expected_clean))

        checked += 1
        beta_base = entry.get("fog", {}).get("beta_base")
        expected_fog = expected_fog_paths(split, city, image, beta_base)
        if fp.get("fog") != expected_fog:
            mismatches.append((image, "fog", fp.get("fog"), expected_fog))

        # rain/snow: file_paths are still null pre-generation in the current
        # schema (their filenames embed rain_rate/snow_rate, unassigned so
        # far) -- nothing to verify against yet, so intentionally skipped
        # rather than flagged.

    print(f"Checked {checked} path fields across {len(manifest)} entries.")
    print(f"Mismatches: {len(mismatches)}")
    for image, field, stored, expected in mismatches[:20]:
        print(f"  {image} [{field}]:")
        print(f"    stored:   {stored}")
        print(f"    expected: {expected}")
    if len(mismatches) > 20:
        print(f"  ... and {len(mismatches) - 20} more")


if __name__ == "__main__":
    main()
