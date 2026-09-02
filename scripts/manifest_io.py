"""
Single source of truth for reading and writing data/dataset_manifest.json.

Every script that MUTATES the manifest -- the stats retrofit, the
shuffled-arm generator, the rain generator, the snow generator, anything
else written later -- MUST go through this module (read_manifest /
atomic_write_manifest / acquire_lock) rather than opening the JSON file
directly with json.load/json.dump. The manifest is a large (multi-MB) file
that many independent scripts will read-modify-write over a period of
weeks; doing that safely needs both an atomic write (so a crash mid-write
can't corrupt or truncate the file) and a lock around the *entire*
read-modify-write cycle (so two scripts running near-simultaneously can't
silently clobber each other's updates -- the classic lost-update race).

Required usage pattern for any script that mutates the manifest:

    from manifest_io import read_manifest, atomic_write_manifest, acquire_lock, DEFAULT_MANIFEST_PATH

    with acquire_lock(DEFAULT_MANIFEST_PATH) as token:
        entries = read_manifest(DEFAULT_MANIFEST_PATH)
        # ... modify entries in place ...
        atomic_write_manifest(entries, DEFAULT_MANIFEST_PATH, lock_token=token)

atomic_write_manifest() REQUIRES lock_token (keyword-only, no default) and
checks it against the currently-active lock acquired via acquire_lock() in
THIS process, raising ManifestLockError if it doesn't match (including if
no lock is held at all). This is enforcement of the pattern above, not
just documentation of it -- but it is NOT bulletproof: it only catches
callers going through this module without holding the lock. A script that
bypasses this module entirely and does its own json.load/json.dump is
still free to corrupt or race the file; nothing here can prevent that.

On-disk format: the manifest file is a JSON OBJECT, not a bare array:
    {"_schema_version": "2.1", "_schema_notes": "...", "entries": [ ... ]}
read_manifest() unwraps this and returns just `entries` (list[dict]);
atomic_write_manifest() takes `entries` (list[dict]) and re-wraps them with
the schema meta fields on write -- callers never touch the wrapper
directly. read_manifest() also tolerates the OLD pre-migration flat-array
format (a bare list, no wrapper).
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"

SCHEMA_VERSION = "2.1"
SCHEMA_NOTES = (
    "Per-arm stats must measure spatial structure, not marginal "
    "distributions. Any scalar aggregate over a permuted quantity (e.g. a "
    "plain mean) is identical between the grounded and shuffled arms by "
    "construction; do not add such stats. Fields prefixed severity_ "
    "(severity_wet_coverage, severity_accumulation_coverage) are dataset "
    "severity/extent descriptors, not ablation-discrimination metrics -- "
    "they are structurally blind to grounded-vs-shuffled for the same "
    "reason. Use the *_depth_corr / *_seg_mi pairs for ablation "
    "discrimination (grounding_depth_corr/grounding_seg_mi for fog and "
    "snow, which have one grounding map each; veiling_/wet_/reflection_"
    "depth_corr and _seg_mi for rain, which has three independently-"
    "grounded components). All mutual information values are in bits "
    "(log2), not nats."
)


class ManifestLockError(RuntimeError):
    """Raised when a manifest lock is already held (by another process, or
    left over from one that crashed without releasing it), or when
    atomic_write_manifest() is called with a missing/stale lock_token."""


# Set by acquire_lock() while its `with` block is active in THIS process;
# checked by atomic_write_manifest(). Cleared back to None on release.
_active_lock_token = None


def read_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> list:
    """Read the manifest and return just the list of entries. Handles both
    the current {"_schema_version":..., "entries":[...]} wrapper and the
    old bare-array format transparently."""
    with open(path) as fh:
        doc = json.load(fh)
    if isinstance(doc, list):
        return doc
    return doc["entries"]


def atomic_write_manifest(entries: list, path: Path = DEFAULT_MANIFEST_PATH, *, lock_token):
    """Write `entries` (list[dict]) to `path`, atomically: write to a temp
    file in the same directory first, then os.replace() -- atomic on
    POSIX, so a crash mid-write leaves either the old file intact or the
    fully-written new one, never a truncated/partial file. Re-wraps
    `entries` with the current schema meta fields on the way out.

    `lock_token` is required (no default) and must match the token yielded
    by the currently-active `acquire_lock()` in this process -- see module
    docstring. Raises ManifestLockError if it doesn't match, including if
    no lock is currently held at all.
    """
    if lock_token is None or lock_token != _active_lock_token:
        raise ManifestLockError(
            "atomic_write_manifest() called without a valid lock_token. "
            "Wrap the read-modify-write cycle in "
            "`with acquire_lock(path) as token:` and pass that token here "
            "(lock_token=token)."
        )

    doc = {
        "_schema_version": SCHEMA_VERSION,
        "_schema_notes": SCHEMA_NOTES,
        "entries": entries,
    }
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp_path, path)


@contextmanager
def acquire_lock(path: Path = DEFAULT_MANIFEST_PATH, timeout: float = 0.0):
    """Context manager: exclusive lock on `path` via a `.lock` sidecar
    file, created with O_EXCL (atomic -- no check-then-create race), AND
    sets the module-level active-token used by atomic_write_manifest()'s
    enforcement check. Yields the token.

    timeout=0 (default): fail immediately with ManifestLockError if the
    lock is already held. timeout>0: poll every 0.1s until acquired or the
    timeout elapses, then raise.
    """
    global _active_lock_token

    lock_path = Path(path).with_name(Path(path).name + ".lock")
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ManifestLockError(
                    f"Manifest lock already held: {lock_path} exists. "
                    f"Another process may be actively reading/writing the "
                    f"manifest, or a previous run crashed without releasing "
                    f"it. If you've confirmed no other process is running, "
                    f"delete {lock_path} manually and retry."
                )
            time.sleep(0.1)

    token = os.urandom(16)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        _active_lock_token = token
        yield token
    finally:
        _active_lock_token = None
        release_lock(path)


def release_lock(path: Path = DEFAULT_MANIFEST_PATH):
    """Remove the `.lock` sidecar file for `path`, if present. Exposed
    standalone (not just via acquire_lock's context-manager exit) for
    manual cleanup of a stale lock left by a crashed process. Does NOT
    clear _active_lock_token on its own -- that's acquire_lock's job on
    exit; calling this manually mid-`with` would desync the two."""
    lock_path = Path(path).with_name(Path(path).name + ".lock")
    lock_path.unlink(missing_ok=True)
