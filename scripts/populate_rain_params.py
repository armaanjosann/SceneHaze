"""
One-shot: populate rain.rain_rate, rain.streak_seed, and rain.file_paths
for all 3,475 manifest entries, and reset rain.<arm>.stats /
snow.<arm>.stats to fresh copies of migrate_manifest.py's
RAIN_STATS_TEMPLATE/SNOW_STATS_TEMPLATE -- all in one lock/atomic-write
cycle, mirroring populate_shuffle_seeds.py's shape.

Stats-reset note: migrate_manifest.py's RAIN_STATS_TEMPLATE and
SNOW_STATS_TEMPLATE ALREADY use the corrected *_ref_corr field names (not
*_seg_mi) -- verified directly during the rain-pipeline audit, no template
edit needed. What's actually stale is the LIVE manifest data: every
entry's rain/snow stats sub-dicts were written by migrate_manifest.py back
when ITS templates still had the old *_seg_mi names (before fog's Session
C fix landed); nothing has touched rain/snow since, since
reset_fog_stats.py only ever reset fog's stats. This script fixes the
live data by replacing each stats sub-dict wholesale with a fresh copy of
the (already-correct) template -- same technique as reset_fog_stats.py,
just extended to rain and snow.

rain.gamma / rain.blue_boost are intentionally NOT populated here and stay
permanently null. Per rain_utils.py's design (see its module docstring),
they're derived at generation time from streak_seed + rain_rate via
sample_atmosphere_shift() and are never stored -- storing them would be
redundant with (and could silently drift from) that derivation.

rain_rate: ONE batched np.random.default_rng(RAIN_GLOBAL_SEED).uniform(10,
150, size=N) draw across all entries in canonical manifest order -- same
pattern as generate_dataset.py's beta_base assignment, avoiding the
per-entry-fresh-rng fragility that pattern exists to avoid. There is no
dedicated "rain_rate_seed" field in the schema (fog's analogous field is
beta_seed) -- RAIN_GLOBAL_SEED is a fixed, documented constant in
rain_utils.py rather than a new manifest field, since nothing currently
reads it back per-entry the way beta_seed is read.

rain.streak_seed: per-entry rain_utils.derive_streak_seed(image) --
deterministic sha256-based, NOT drawn from the batched rng stream above
(see rain_utils.py's module docstring for why atmosphere/streak seeds are
independently derived rather than sharing one stream).

rain.file_paths: computed via migrate_manifest.rain_file_paths() now that
rain_rate is known, filling in what has been `null` since the schema
migration -- generate_rain_batch.py needs these paths to know where to
write.

Usage:
  python3 scripts/populate_rain_params.py
"""

import numpy as np

from manifest_io import read_manifest, atomic_write_manifest, acquire_lock, DEFAULT_MANIFEST_PATH
from migrate_manifest import RAIN_STATS_TEMPLATE, SNOW_STATS_TEMPLATE, rain_file_paths
from rain_utils import RAIN_GLOBAL_SEED, derive_streak_seed

RAIN_RATE_RANGE = (10.0, 150.0)
ARMS = ["constant", "grounded", "shuffled"]


def main():
    with acquire_lock(DEFAULT_MANIFEST_PATH) as token:
        manifest = read_manifest(DEFAULT_MANIFEST_PATH)

        rng = np.random.default_rng(RAIN_GLOBAL_SEED)
        rain_rates = rng.uniform(RAIN_RATE_RANGE[0], RAIN_RATE_RANGE[1], size=len(manifest))

        for entry, rain_rate in zip(manifest, rain_rates):
            rain_rate = round(float(rain_rate), 4)
            entry["rain"]["rain_rate"] = rain_rate
            entry["rain"]["streak_seed"] = derive_streak_seed(entry["image"])
            entry["file_paths"]["rain"] = rain_file_paths(entry["split"], entry["city"], entry["image"], rain_rate)

            for arm in ARMS:
                entry["rain"][arm]["stats"] = dict(RAIN_STATS_TEMPLATE)
                entry["snow"][arm]["stats"] = dict(SNOW_STATS_TEMPLATE)

        atomic_write_manifest(manifest, DEFAULT_MANIFEST_PATH, lock_token=token)

    print(f"Set rain_rate + streak_seed + file_paths.rain for {len(manifest)} entries.")
    print(f"Reset rain.<arm>.stats and snow.<arm>.stats to fresh templates for {len(manifest)} entries x {len(ARMS)} arms.")
    sample = manifest[0]
    print(f"Sample: {sample['image']} -> rain_rate={sample['rain']['rain_rate']}, "
          f"streak_seed={sample['rain']['streak_seed']}")
    print(f"  file_paths.rain.grounded.rgb = {sample['file_paths']['rain']['grounded']['rgb']}")


if __name__ == "__main__":
    main()
