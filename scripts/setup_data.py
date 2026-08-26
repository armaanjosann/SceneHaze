"""
Organize downloaded Cityscapes data into the SceneHaze project structure.

We symlink rather than copy: leftImg8bit (~10GB) + gtFine (~800MB) already sit
on Desktop, and disk space is tight. Symlinking per-city keeps the project
structure clean (data/clean/cityscapes/<split>/<city>/) without duplicating
data on disk.

Safe to re-run: skips any link that already exists and points to the right place.
"""

import os
from pathlib import Path

# Source locations (raw extracted zips)
LEFTIMG_SRC = Path.home() / "Desktop" / "leftImg8bit_trainvaltest" / "leftImg8bit"
GTFINE_SRC = Path.home() / "Desktop" / "gtFine_trainvaltest (1)" / "gtFine"

# Destination locations (inside project)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DST = PROJECT_ROOT / "data" / "clean" / "cityscapes"
SEG_DST = PROJECT_ROOT / "data" / "segmentation" / "cityscapes"

SPLITS = ["train", "val", "test"]


def link_split(src_root: Path, dst_root: Path, label: str):
    if not src_root.exists():
        print(f"  [SKIP] source not found: {src_root}")
        return

    for split in SPLITS:
        src_split = src_root / split
        if not src_split.exists():
            print(f"  [SKIP] {label}/{split}: no source dir")
            continue

        dst_split = dst_root / split
        dst_split.mkdir(parents=True, exist_ok=True)

        cities = sorted(p.name for p in src_split.iterdir() if p.is_dir())
        for city in cities:
            src_city = src_split / city
            dst_city = dst_split / city

            if dst_city.is_symlink():
                if dst_city.resolve() == src_city.resolve():
                    continue  # already linked correctly
                dst_city.unlink()
            elif dst_city.exists():
                print(f"  [WARN] {dst_city} exists and is not a symlink — leaving it alone")
                continue

            dst_city.symlink_to(src_city, target_is_directory=True)

        print(f"  [OK] {label}/{split}: linked {len(cities)} cities")


def main():
    print("Linking clean images (leftImg8bit -> data/clean/cityscapes)...")
    link_split(LEFTIMG_SRC, CLEAN_DST, "clean")

    print("\nLinking segmentation labels (gtFine -> data/segmentation/cityscapes)...")
    link_split(GTFINE_SRC, SEG_DST, "segmentation")

    # Sanity check: count files. Note pathlib's rglob does not follow
    # symlinked directories, so we walk manually with followlinks=True.
    def count_files(root: Path, suffix: str) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            total += sum(1 for f in filenames if f.endswith(suffix))
        return total

    n_clean = count_files(CLEAN_DST, ".png")
    n_seg = count_files(SEG_DST, "_labelIds.png")
    print(f"\nTotal clean images linked: {n_clean}")
    print(f"Total labelIds masks linked: {n_seg}")


if __name__ == "__main__":
    main()
