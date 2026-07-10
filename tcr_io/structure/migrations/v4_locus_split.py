"""v3 -> v4: per-locus repertoire layout (TR/BCR chain support, Phase 1).

Pre-v4 datasets are TRB-only with a flat layout:
  - ``processed_repertoires/*.parquet``          (one file per repertoire, no locus)
  - ``meta/repertoire/repertoire.parquet``       (single repertoire-meta table)

v4 makes locus the directory segment and stores repertoire meta one parquet per locus
(Option C). Since all existing data is TRB, the migration just **relocates in place** — no
column is added or rewritten (locus lives in the path, not the frame):
  - ``processed_repertoires/*.parquet``    -> ``processed_repertoires/TRB/*.parquet``
  - ``meta/repertoire/repertoire.parquet`` -> ``meta/repertoire/TRB.parquet``

Data-preserving (no re-ingest): processed repertoires are ingested source, not a derivable
operation output, so they cannot be cleared-and-regenerated. Old paths are hardcoded here
because a migration runs on an *older* tree than the current layout describes.
"""
import shutil

from .base import migration
from ..layout import Layout, repertoire_meta_relpath


@migration(to_version=4, description="Split processed repertoires into per-locus subdirs (existing data is TRB)")
def upgrade(ds):
    processed = ds.db_dir / Layout.processed_dir.path
    trb = processed / "TRB"
    trb.mkdir(parents=True, exist_ok=True)
    for f in processed.glob("*.parquet"):     # only flat top-level files; the new TRB/ dir is skipped
        shutil.move(str(f), str(trb / f.name))

    old_meta = ds.db_dir / "meta/repertoire/repertoire.parquet"   # the pre-v4 single meta table
    if old_meta.exists():
        shutil.move(str(old_meta), str(ds.db_dir / repertoire_meta_relpath("TRB")))
