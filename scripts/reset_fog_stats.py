"""
One-shot: null out fog.{constant,grounded,shuffled}.stats for all 3,475
manifest entries, replacing the whole stats sub-dict with a fresh
all-null FOG_STATS_TEMPLATE (imported from migrate_manifest.py, not
redefined here, so there's one source of truth for the field list).

Needed because the schema changed (grounding_seg_mi -> grounding_ref_corr,
see manifest_io.SCHEMA_NOTES and the Session C commit that made this
change) -- a plain "set existing values to None" would leave the OLD
grounding_seg_mi key sitting in every entry alongside a newly-added
grounding_ref_corr key. Replacing the whole sub-dict removes the stale
key cleanly.

Does NOT touch fog.beta_base, fog.beta_seed, or fog.shuffle_seed --
only the stats sub-dicts.

Usage:
  python3 scripts/reset_fog_stats.py
"""

from migrate_manifest import FOG_STATS_TEMPLATE
from manifest_io import read_manifest, atomic_write_manifest, acquire_lock, DEFAULT_MANIFEST_PATH

ARMS = ["constant", "grounded", "shuffled"]


def main():
    with acquire_lock(DEFAULT_MANIFEST_PATH) as token:
        manifest = read_manifest(DEFAULT_MANIFEST_PATH)

        for entry in manifest:
            for arm in ARMS:
                entry["fog"][arm]["stats"] = dict(FOG_STATS_TEMPLATE)

        atomic_write_manifest(manifest, DEFAULT_MANIFEST_PATH, lock_token=token)

    print(f"Reset stats for {len(manifest)} entries x {len(ARMS)} arms.")
    print(f"Fresh stats shape: {FOG_STATS_TEMPLATE}")


if __name__ == "__main__":
    main()
