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
no re-ingest.

Every path here is a v4-era literal, and the manifest is read-modified-written as raw JSON
rather than through ``Manifest`` — a later manifest field must survive this step untouched,
and a later *renamed* one must not silently reshape a v3 tree. See base.py.
"""
import json
import shutil
from pathlib import Path

from .base import migration

_V4_PROCESSED_DIR = "processed_repertoires"
_V4_MANIFEST      = "meta/manifest.json"


def _v4_repertoire_locus_relpath(rep_id: str, locus: str) -> str:
    # v3 filenames are already filesystem-safe stems, so no re-sanitising is needed here.
    return f"{_V4_PROCESSED_DIR}/{rep_id}/locus={locus}/{rep_id}.parquet"


@migration(to_version=4, description="Relocate TRB repertoires into the per-repertoire hive layout")
def upgrade(db_dir: Path) -> None:
    processed = db_dir / _V4_PROCESSED_DIR
    for f in sorted(processed.glob("*.parquet")):   # flat v3 files only (new {id}/ dirs are skipped)
        rep_id = f.stem                             # v3 wrote {safe_id}.parquet
        dest = db_dir / _v4_repertoire_locus_relpath(rep_id, "TRB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))

    old_meta = db_dir / "meta/repertoire/repertoire.parquet"   # the pre-v4 single meta table
    if old_meta.exists():
        shutil.move(str(old_meta), str(db_dir / "meta/repertoire/TRB.parquet"))

    # Record present loci (fixed at ingest going forward). Read-modify-write on the raw JSON;
    # migrate()'s _write_version preserves present_loci when it bumps the version afterwards.
    man_path = db_dir / _V4_MANIFEST
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man["present_loci"] = ["TRB"]
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(man, indent=2))
