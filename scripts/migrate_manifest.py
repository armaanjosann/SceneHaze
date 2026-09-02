"""
Migrate data/dataset_manifest.json (flat, fog-only schema) to the new
nested multi-weather schema, written to data/dataset_manifest_v2.json (the
ORIGINAL is never touched by this script -- rename manually after
verifying the output).

One entry per image (same 3,475 as today), identity fields (split/city/
image) at top level, then per-weather-type sub-objects (fog/rain/snow),
each with its driving parameters + seeds and per-arm (constant/grounded/
shuffled) stats sub-objects, plus a provenance block (generation
timestamps only -- seeds live in their weather sub-object) and a
file_paths block.

This migration only RESHAPES what already exists: fog.beta_base (migrated
verbatim), fog.beta_seed (the known global seed=42 used by
generate_dataset.py::build_manifest), and file_paths for fog's constant/
grounded arms (computed from the existing, unchanged naming convention).
Everything else is explicitly null. This script does not compute stats,
does not generate images, and does not invent values for anything it
can't derive from the current manifest + current on-disk naming
convention.

Stats shape differs per weather type (see manifest_io.SCHEMA_NOTES for the
reasoning): fog and snow each have ONE grounding map, so they get a single
grounding_depth_corr/grounding_ref_corr pair. Rain has THREE independently-
grounded components (veiling beta, wet-darkening mask, reflection mask),
so it gets three separate depth_corr/ref_corr pairs instead. Originally
these used a *_seg_mi mutual-information field instead of *_ref_corr, but
MI is invariant to bijective relabeling of either variable -- it can't
distinguish a correctly-grounded map from a shuffled one that permutes
which value lands on which category, since both are equally "informative"
about segmentation. Replaced with ref_corr (Pearson correlation against
the canonical grounded map), which IS sensitive to the actual values, not
just partition informativeness. severity_wet_coverage /
severity_accumulation_coverage are explicitly named severity_* (not
mean_*) to make clear they're dataset descriptors, not ablation-
discrimination stats -- they're structurally blind to grounded-vs-shuffled
for the same reason a plain mean_beta would be (see SCHEMA_NOTES).

rain/snow file_paths are left null (not schema-shaped-with-nulls) because
their filenames embed rain_rate/snow_rate, which don't exist yet.

fog.shuffled's path points to a NEW sibling folder (scene_grounded_shuffled/)
rather than any renamed existing folder.

Uses manifest_io.atomic_write_manifest() / read_manifest() / acquire_lock()
per manifest_io's module contract -- including taking a lock on the v2
path even though nothing else can be concurrently writing to a file this
script is creating fresh. Done purely for API consistency (every script
that writes a manifest file goes through the same acquire-lock-then-write
pattern, no exceptions to remember), not because it's protecting against a
real race here.
"""

from pathlib import Path

from manifest_io import read_manifest, atomic_write_manifest, acquire_lock, DEFAULT_MANIFEST_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLD_MANIFEST_PATH = DEFAULT_MANIFEST_PATH  # data/dataset_manifest.json
NEW_MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest_v2.json"

CLEAN_ROOT = PROJECT_ROOT / "data" / "clean" / "cityscapes"
CONSTANT_ROOT = PROJECT_ROOT / "data" / "generated" / "constant_beta"
GROUNDED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded"
SHUFFLED_ROOT = PROJECT_ROOT / "data" / "generated" / "scene_grounded_shuffled"  # net-new arm, folder doesn't exist yet

RAIN_CONSTANT_ROOT = PROJECT_ROOT / "data" / "generated" / "rain_constant"
RAIN_GROUNDED_ROOT = PROJECT_ROOT / "data" / "generated" / "rain_grounded"
RAIN_SHUFFLED_ROOT = PROJECT_ROOT / "data" / "generated" / "rain_shuffled"  # arm not implemented (deferred to
# extensions, see rain design proposal) -- path predicted for schema
# consistency only, same as fog's shuffled path existed before shuffled
# generation happened.

EXPECTED_ENTRY_COUNT = 3475
ORIGINAL_BETA_SEED = 42  # generate_dataset.py::build_manifest's seed, at time of original generation

FOG_STATS_TEMPLATE = {
    "mean_abs_delta": None,
    "grounding_depth_corr": None,
    "grounding_ref_corr": None,
    "mean_transmission": None,
}
RAIN_STATS_TEMPLATE = {
    "mean_abs_delta": None,
    "veiling_depth_corr": None,
    "veiling_ref_corr": None,
    "wet_depth_corr": None,
    "wet_ref_corr": None,
    "reflection_depth_corr": None,
    "reflection_ref_corr": None,
    "severity_wet_coverage": None,
    "mean_veiling_transmission": None,
}
SNOW_STATS_TEMPLATE = {
    "mean_abs_delta": None,
    "grounding_depth_corr": None,
    "grounding_ref_corr": None,
    "severity_accumulation_coverage": None,
}


def empty_arm(stats_template: dict) -> dict:
    return {"stats": dict(stats_template)}


def fog_file_paths(split: str, city: str, image: str, beta_base: float) -> dict:
    """Predictable paths for fog's three arms. constant/grounded RGB paths
    follow the EXISTING naming convention unchanged (matches what
    generate_dataset.py actually wrote to disk). The _aux.npz paths for all
    three arms are NEW -- nothing currently saves raw beta/transmission
    arrays (generate_grounded.py only saves a normalized-for-viewing
    _betamap.png, which can't back grounding_depth_corr/mean_transmission
    since it's been rescaled per-image and lost the true beta scale)."""
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


def rain_file_paths(split: str, city: str, image: str, rain_rate: float) -> dict:
    """Predictable paths for rain's three arms, analogous to
    fog_file_paths above but keyed on rain_rate instead of beta_base.
    1 decimal place (rain_rate spans 10-150; fog's 2dp precision, needed
    for its 0.4-2.0 range, would be overkill here). Called from
    populate_rain_params.py once rain_rate is assigned -- unlike fog's
    original migration, rain_rate is known before generation happens, so
    these paths are computable immediately rather than staying null."""
    r = f"{rain_rate:.1f}"

    constant_stem = f"{image}_rain{r}_constant"
    grounded_stem = f"{image}_rain{r}_grounded"
    shuffled_stem = f"{image}_rain{r}_shuffled"

    return {
        "constant": {
            "rgb": str((RAIN_CONSTANT_ROOT / split / city / f"{constant_stem}.png").relative_to(PROJECT_ROOT)),
            "aux": str((RAIN_CONSTANT_ROOT / split / city / f"{constant_stem}_aux.npz").relative_to(PROJECT_ROOT)),
        },
        "grounded": {
            "rgb": str((RAIN_GROUNDED_ROOT / split / city / f"{grounded_stem}.png").relative_to(PROJECT_ROOT)),
            "aux": str((RAIN_GROUNDED_ROOT / split / city / f"{grounded_stem}_aux.npz").relative_to(PROJECT_ROOT)),
        },
        "shuffled": {
            "rgb": str((RAIN_SHUFFLED_ROOT / split / city / f"{shuffled_stem}.png").relative_to(PROJECT_ROOT)),
            "aux": str((RAIN_SHUFFLED_ROOT / split / city / f"{shuffled_stem}_aux.npz").relative_to(PROJECT_ROOT)),
        },
    }


def migrate_entry(old: dict) -> dict:
    split, city, image, beta_base = old["split"], old["city"], old["image"], old["beta_base"]

    return {
        "split": split,
        "city": city,
        "image": image,

        "fog": {
            "beta_base": beta_base,
            "beta_seed": ORIGINAL_BETA_SEED,
            "shuffle_seed": None,
            "constant": empty_arm(FOG_STATS_TEMPLATE),
            "grounded": empty_arm(FOG_STATS_TEMPLATE),
            "shuffled": empty_arm(FOG_STATS_TEMPLATE),
        },

        "rain": {
            "rain_rate": None, "gamma": None, "blue_boost": None, "streak_seed": None,
            "constant": empty_arm(RAIN_STATS_TEMPLATE),
            "grounded": empty_arm(RAIN_STATS_TEMPLATE),
            "shuffled": empty_arm(RAIN_STATS_TEMPLATE),
        },

        "snow": {
            "snow_rate": None, "accumulation_strength": None, "particle_seed": None,
            "constant": empty_arm(SNOW_STATS_TEMPLATE),
            "grounded": empty_arm(SNOW_STATS_TEMPLATE),
            "shuffled": empty_arm(SNOW_STATS_TEMPLATE),
        },

        "provenance": {
            "generation_timestamps": {
                "fog_constant": None, "fog_grounded": None, "fog_shuffled": None,
                "rain_constant": None, "rain_grounded": None, "rain_shuffled": None,
                "snow_constant": None, "snow_grounded": None, "snow_shuffled": None,
            },
        },

        "file_paths": {
            "clean": str((CLEAN_ROOT / split / city / f"{image}_leftImg8bit.png").relative_to(PROJECT_ROOT)),
            "fog": fog_file_paths(split, city, image, beta_base),
            "rain": None,  # not computable until rain_rate is assigned
            "snow": None,  # not computable until snow_rate is assigned
        },
    }


def main():
    if not OLD_MANIFEST_PATH.exists():
        raise SystemExit(f"Original manifest not found: {OLD_MANIFEST_PATH}")

    size_before = OLD_MANIFEST_PATH.stat().st_size
    old_entries = read_manifest(OLD_MANIFEST_PATH)  # tolerates the old bare-array format

    new_entries = []
    field_anomalies = []
    missing_constant_files = []
    missing_grounded_files = []

    for i, old_entry in enumerate(old_entries):
        for key in ("split", "city", "image", "beta_base"):
            if key not in old_entry or old_entry[key] is None:
                field_anomalies.append(f"entry {i} ({old_entry.get('image', '?')}): missing/null '{key}'")

        new_entry = migrate_entry(old_entry)
        new_entries.append(new_entry)

        constant_rgb = PROJECT_ROOT / new_entry["file_paths"]["fog"]["constant"]["rgb"]
        grounded_rgb = PROJECT_ROOT / new_entry["file_paths"]["fog"]["grounded"]["rgb"]
        if not constant_rgb.exists():
            missing_constant_files.append(new_entry["image"])
        if not grounded_rgb.exists():
            missing_grounded_files.append(new_entry["image"])

    beta_mismatches = [
        (old_e["image"], old_e["beta_base"], new_e["fog"]["beta_base"])
        for old_e, new_e in zip(old_entries, new_entries)
        if old_e["beta_base"] != new_e["fog"]["beta_base"]
    ]

    with acquire_lock(NEW_MANIFEST_PATH) as token:
        atomic_write_manifest(new_entries, NEW_MANIFEST_PATH, lock_token=token)

    size_after = NEW_MANIFEST_PATH.stat().st_size

    reloaded = read_manifest(NEW_MANIFEST_PATH)
    round_trips_ok = len(reloaded) == len(new_entries)

    print("=== Migration summary ===")
    print(f"Original manifest: {OLD_MANIFEST_PATH}")
    print(f"New manifest:      {NEW_MANIFEST_PATH}  (original untouched)")
    print()
    count_ok = len(old_entries) == len(new_entries) == EXPECTED_ENTRY_COUNT
    print(f"Entries: {len(old_entries)} -> {len(new_entries)} "
          f"({'OK' if count_ok else f'MISMATCH -- expected {EXPECTED_ENTRY_COUNT}'})")
    print(f"Size: {size_before:,} bytes -> {size_after:,} bytes "
          f"({size_after / max(size_before, 1):.1f}x larger)")
    print(f"JSON round-trips correctly: {round_trips_ok}")
    print()
    print(f"beta_base data loss: {len(beta_mismatches)} mismatches"
          + (f" -- e.g. {beta_mismatches[:5]}" if beta_mismatches else " (none)"))
    print(f"Field-validation anomalies: {len(field_anomalies)}"
          + (f" -- first 5: {field_anomalies[:5]}" if field_anomalies else " (none)"))
    print(f"Predicted constant PNG missing on disk: {len(missing_constant_files)}"
          + (f" -- first 5: {missing_constant_files[:5]}" if missing_constant_files else " (none, all present)"))
    print(f"Predicted grounded PNG missing on disk: {len(missing_grounded_files)}"
          + (f" -- first 5: {missing_grounded_files[:5]}" if missing_grounded_files else " (none, all present)"))
    print()
    print("Original file NOT modified. After verifying the numbers above, rename manually:")
    print(f"  cd {OLD_MANIFEST_PATH.parent}")
    print(f"  mv {NEW_MANIFEST_PATH.name} {OLD_MANIFEST_PATH.name}")


if __name__ == "__main__":
    main()
