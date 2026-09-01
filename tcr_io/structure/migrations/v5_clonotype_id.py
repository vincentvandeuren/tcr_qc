"""v4 -> v5: backfill a per-(repertoire, locus) ``clonotype_id`` into each hive parquet.

v4 processed files are ``processed_repertoires/{id}/locus={LOCUS}/{id}.parquet`` with no
``clonotype_id`` column. v5 adds it as the leading column: a row number **within each locus
partition** — which is exactly what a per-file `with_row_index` produces, so the backfill is
one independent rewrite per file (no cross-file coordination). This matches the ingest-time
`int_range().over("locus")` numbering by construction.

Single-cell datasets don't exist pre-v5 (the `cell_id`/`clone_to_cell` machinery ships with v5),
so there is nothing to split here — only the bulk `clonotype_id` backfill.

Both the glob and the column list are v5-era literals. Neither may be taken from the live
layout or schema: the glob describes a *v4* tree, and the select must produce exactly the
*v5* column set — a later schema edit reaching in here would rewrite history. See base.py.
"""
from pathlib import Path

import polars as pl

from .base import migration

_V4_LOCI_GLOB = "processed_repertoires/*/locus=*/*.parquet"

# The v5 repertoire schema, in order. clonotype_id leads; dtypes are fixed by the cast above it.
_V5_REPERTOIRE_COLUMNS = [
    "clonotype_id",
    "repertoire_id",
    "junction",
    "v_call",
    "junction_aa",
    "j_call",
    "duplicate_count",
    "filter_pass",
]


@migration(to_version=5, description="Backfill per-(repertoire, locus) clonotype_id into processed parquets")
def upgrade(db_dir: Path) -> None:
    for f in sorted(db_dir.glob(_V4_LOCI_GLOB)):
        tmp = f.with_suffix(".tmp.parquet")
        (pl.scan_parquet(f)
           .with_row_index("clonotype_id")
           .with_columns(pl.col("clonotype_id").cast(pl.UInt32))
           .select(_V5_REPERTOIRE_COLUMNS)
           .sink_parquet(tmp))
        tmp.replace(f)
