"""v0 -> v1: backfill the version manifest for a pre-versioning dataset.

The on-disk tree already matches v1 (versioning is additive), so nothing moves. This
migration exists so a manifest-less dataset gets stamped; the driver writes
`Manifest(version=1)` after this returns.
"""
from .base import migration
from ..layout import REQUIRED_DIRS


@migration(to_version=1, description="Backfill version manifest for a pre-versioning dataset")
def upgrade(ds):
    missing = [d for d in REQUIRED_DIRS if not (ds.db_dir / d).exists()]
    if missing:
        raise RuntimeError(f"Cannot migrate to v1: not a v1-shaped dataset (missing {missing})")
    # nothing to move; driver commits Manifest(version=1) after this returns.
