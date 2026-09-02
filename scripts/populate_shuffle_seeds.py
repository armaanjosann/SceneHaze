"""
One-shot: populate fog.shuffle_seed for all 3,475 manifest entries.

Pure computation (sha256 of image name + global seed, see
generate_shuffled.py::derive_shuffle_seed) -- no I/O beyond the manifest
read/write itself, so this is fast (~1s) despite touching every entry.
Single lock/read/modify/write cycle via manifest_io, not one per entry.

Usage:
  python3 scripts/populate_shuffle_seeds.py
"""

from generate_shuffled import derive_shuffle_seed
from manifest_io import read_manifest, atomic_write_manifest, acquire_lock, DEFAULT_MANIFEST_PATH


def main():
    with acquire_lock(DEFAULT_MANIFEST_PATH) as token:
        manifest = read_manifest(DEFAULT_MANIFEST_PATH)

        n_set = 0
        for entry in manifest:
            entry["fog"]["shuffle_seed"] = derive_shuffle_seed(entry["image"])
            n_set += 1

        atomic_write_manifest(manifest, DEFAULT_MANIFEST_PATH, lock_token=token)

    print(f"Set shuffle_seed for {n_set} entries.")
    print(f"Sample: {manifest[0]['image']} -> {manifest[0]['fog']['shuffle_seed']}")


if __name__ == "__main__":
    main()
