"""v4 -> v5: backfill a per-(repertoire, locus) ``clonotype_id`` into each hive parquet.

v4 processed files are ``processed_repertoires/{id}/locus={LOCUS}/{id}.parquet`` with no
``clonotype_id`` column. v5 adds it as the leading column: a row number **within each locus
partition** — which is exactly what a per-file `with_row_index` produces, so the backfill is
one independent rewrite per file (no cross-file coordination). This matches the ingest-time
`int_range().over("locus")` numbering by construction.

Single-cell datasets don't exist pre-v5 (the `cell_id`/`clone_to_cell` machinery ships with v5),
so there is nothing to split here — only the bulk `clonotype_id` backfill.

Runs on raw paths (never the current-schema accessors), per the migration contract.
"""
import polars as pl

from .base import migration
from ..layout import loci_glob
from ..schema import REPERTOIRE


@migration(to_version=5, description="Backfill per-(repertoire, locus) clonotype_id into processed parquets")
def upgrade(ds):
    for f in sorted(ds.db_dir.glob(loci_glob())):     # processed_repertoires/*/locus=*/*.parquet
        tmp = f.with_suffix(".tmp.parquet")
        (pl.scan_parquet(f)
           .with_row_index("clonotype_id")
           .with_columns(pl.col("clonotype_id").cast(pl.UInt32))
           .select(REPERTOIRE.keys())                 # clonotype_id first; dtypes fixed
           .sink_parquet(tmp))
        tmp.replace(f)
