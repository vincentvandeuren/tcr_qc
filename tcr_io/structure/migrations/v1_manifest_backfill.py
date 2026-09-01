"""v0 -> v1: backfill the version manifest for a pre-versioning dataset.

The on-disk tree already matches v1 (versioning is additive), so nothing moves. This
migration exists so a manifest-less dataset gets stamped; the driver writes
`Manifest(version=1)` after this returns.
"""
from pathlib import Path

from .base import migration

# The v1-era required dirs, pinned. Not `layout.REQUIRED_DIRS` — see base.py.
_V1_REQUIRED_DIRS = [
    "processed_repertoires",
    "meta",
    "meta/repertoire",
    "meta/patient",
    "meta/publication",
]


@migration(to_version=1, description="Backfill version manifest for a pre-versioning dataset")
def upgrade(db_dir: Path) -> None:
    missing = [d for d in _V1_REQUIRED_DIRS if not (db_dir / d).exists()]
    if missing:
        raise RuntimeError(f"Cannot migrate to v1: not a v1-shaped dataset (missing {missing})")
    # nothing to move; driver commits Manifest(version=1) after this returns.
