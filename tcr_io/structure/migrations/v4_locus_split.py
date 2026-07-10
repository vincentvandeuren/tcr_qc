"""v3 -> v4: per-repertoire hive layout (TR/BCR chain support).

Pre-v4 datasets are TRB-only with a flat layout:
  - ``processed_repertoires/{id}.parquet``       (one file per repertoire, no locus)
  - ``meta/repertoire/repertoire.parquet``       (single repertoire-meta table)

v4 makes locus a hive partition **under each repertoire's directory**, written in one
`pl.PartitionBy` pass at ingest:
  - ``processed_repertoires/{id}.parquet``    -> ``processed_repertoires/{id}/locus=TRB/{id}.parquet``
  - ``meta/repertoire/repertoire.parquet``    -> ``meta/repertoire/TRB.parquet``
and records ``present_loci=["TRB"]`` in the manifest.

Since `locus` lives only in the path (``include_key=False``), the flat v3 files already have the
right columns — the migration just **moves** them into the hive path (no rewrite). Data-preserving;
no re-ingest. Old paths are hardcoded because a migration runs on an *older* tree than the current
layout describes.
"""
import shutil

from .base import migration
from ..layout import Layout, repertoire_locus_relpath, repertoire_meta_relpath
from ..version import Manifest


@migration(to_version=4, description="Relocate TRB repertoires into the per-repertoire hive layout")
def upgrade(ds):
    processed = ds.db_dir / Layout.processed_dir.path
    for f in list(processed.glob("*.parquet")):     # flat v3 files only (new {id}/ dirs are skipped)
        rep_id = f.stem                             # v3 wrote {safe_id}.parquet
        dest = ds.db_dir / repertoire_locus_relpath(rep_id, "TRB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))

    old_meta = ds.db_dir / "meta/repertoire/repertoire.parquet"   # the pre-v4 single meta table
    if old_meta.exists():
        shutil.move(str(old_meta), str(ds.db_dir / repertoire_meta_relpath("TRB")))

    # Record present loci (fixed at ingest going forward). Read-modify-write; migrate()'s
    # _write_version preserves present_loci when it bumps the version afterwards.
    man_path = ds.db_dir / Layout.manifest.path
    man = Manifest.read(man_path)
    Manifest(version=man.version, tcrio_version=man.tcrio_version, present_loci=["TRB"]).write(man_path)
