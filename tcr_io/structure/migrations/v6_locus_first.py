"""v5 -> v6: locus-first shards, repertoire meta split by grain, `filter_pass` -> `filter_reason`.

A v5 tree nests the locus *under* the repertoire and carries one meta table per locus:

  - ``processed_repertoires/{id}/locus={L}/{id}.parquet``
  - ``meta/repertoire/{L}.parquet``   (repertoire_id | source_files | patient_id |
                                       n_clonotypes | n_filtered_clonotypes | total_duplicates)
  - ``meta/repertoire/repertoire_meta.parquet``, ``meta/patient/patient_meta.parquet``
  - ``meta/generation.json``, ``meta/publication/publication_ids.json``  (both NDJSON content)
  - ``meta/manifest.json`` with a ``present_loci`` list

v6 inverts the first level and splits the second by grain:

  - ``processed_repertoires/locus={L}/{id}.parquet`` — one listing per locus, and the set of
    loci a dataset holds becomes a directory read instead of a manifest field, so
    ``present_loci`` is dropped rather than migrated.
  - ``meta/repertoire/repertoire.parquet`` (locus-invariant, deduped) +
    ``meta/repertoire/locus={L}/counts.parquet`` (per-locus counts).
  - side tables renamed to ``extra.parquet`` under the grain they extend; the two NDJSON files
    renamed off ``.json``.

**`filter_pass` -> `filter_reason`.** v6 stores *why* a row was excluded, and v5 stored only
that it was. The reason is not recoverable: reconstructing it would mean importing live filter
code and the current IMGT reference to re-evaluate masks against data filtered by an older
version of both — the drift this package's doctrine exists to prevent. So every excluded row
gets one honest placeholder, ``"unknown"``, which `FilteringReport` counts like any other
reason and describes as exactly what it is. Rows under ``locus=_unassigned`` are stamped
excluded whatever their old boolean said: that partition holds rows no locus would take, and
the v6 invariant is ``passed <=> filter_reason.is_null()``.

``operations/`` is deleted. Its records key on an output set that changed shape in this bump,
and everything under it is regenerable by definition — clearing beats migrating a cache.

Every path, glob, column list and dtype below is a v6-era literal. None may come from the live
layout or schema: they describe a *v5* tree on the way to a *v6* one, both frozen. See base.py.
"""
import json
import shutil
import warnings
from pathlib import Path

import polars as pl

from .base import migration

_PROCESSED = "processed_repertoires"
_V5_SHARDS = f"{_PROCESSED}/*/locus=*/*.parquet"     # {id}/locus={L}/{id}.parquet
_V6_SHARDS = f"{_PROCESSED}/locus=*/*.parquet"       # locus={L}/{id}.parquet

_META_REPERTOIRE = "meta/repertoire"
_UNASSIGNED = "_unassigned"

# The loci a v5 tree could name a meta table after. Pinned, because `meta/repertoire/` also
# holds files that are NOT per-locus tables (the side table, and after this step the two new
# ones), and a bare `*.parquet` glob would swallow them.
_V5_META_LOCI = ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL", _UNASSIGNED)

# The v6 repertoire schema, in order. Same as v5 with the last column replaced.
_V6_REPERTOIRE_COLUMNS = [
    "clonotype_id",
    "repertoire_id",
    "junction",
    "v_call",
    "junction_aa",
    "j_call",
    "duplicate_count",
    "filter_reason",
]

# The two halves the single v5 meta table splits into.
_V6_META_COLUMNS = {
    "repertoire_id": pl.Utf8,
    "source_files": pl.List(pl.Utf8),
    "patient_id": pl.Utf8,
}
_V6_COUNTS_COLUMNS = {
    "repertoire_id": pl.Utf8,
    "n_clonotypes": pl.Int64,
    "n_filtered_clonotypes": pl.Int64,
    "total_duplicates": pl.Int64,
}

# The placeholder reason. Not a filter name: no filter in any version produces it, which is
# what makes it readable as "this row predates per-row reasons".
_UNRECOVERABLE = "unknown"

# src -> dst, both v6-era literals. NDJSON content that was named `.json`, and side tables
# renamed to say what they are rather than repeating their directory.
_RENAMES = {
    f"{_META_REPERTOIRE}/repertoire_meta.parquet": f"{_META_REPERTOIRE}/extra.parquet",
    "meta/patient/patient_meta.parquet": "meta/patient/extra.parquet",
    "meta/generation.json": "meta/generation.ndjson",
    "meta/publication/publication_ids.json": "meta/publication/publication_ids.ndjson",
}


@migration(to_version=6,
           description="Locus-first shards, repertoire meta split by grain, filter_pass -> filter_reason")
def upgrade(db_dir: Path) -> None:
    _relocate_shards(db_dir)
    _rewrite_filter_reason(db_dir)
    _split_repertoire_meta(db_dir)
    _rename_files(db_dir)
    _clear_operations(db_dir)
    _drop_present_loci(db_dir)


def _relocate_shards(db_dir: Path) -> None:
    """`{id}/locus={L}/{id}.parquet` -> `locus={L}/{id}.parquet`, then drop the empty shells.

    The v5 glob is three segments deep and the v6 path is two, so a re-run matches nothing:
    the move is its own idempotence. A leftover `{id}/` directory is removed only when nothing
    is left under it, so an interrupted move cannot lose a shard it had not relocated yet.
    """
    processed = db_dir / _PROCESSED
    for f in sorted(db_dir.glob(_V5_SHARDS)):
        dest = processed / f.parent.name / f.name        # parent is `locus={L}`
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))

    if not processed.is_dir():
        return
    for d in sorted(processed.iterdir()):
        if d.is_dir() and not d.name.startswith("locus=") and not any(d.rglob("*.parquet")):
            shutil.rmtree(d)


def _rewrite_filter_reason(db_dir: Path) -> None:
    """Replace the boolean with the reason column, one independent rewrite per shard.

    A shard that already has `filter_reason` is left alone, so a re-run after an interrupt
    finishes the remainder instead of double-rewriting what it did. A null `filter_pass`
    counts as excluded: v5 reads selected rows with `filter(col("filter_pass"))`, which drops
    nulls, so that is what the data already meant.
    """
    for f in sorted(db_dir.glob(_V6_SHARDS)):
        scan = pl.scan_parquet(f)
        if "filter_reason" in scan.collect_schema().names():
            continue
        excluded = (pl.lit(True) if f.parent.name == f"locus={_UNASSIGNED}"
                    else ~pl.col("filter_pass").fill_null(False))
        tmp = f.with_suffix(".tmp.parquet")
        (scan.with_columns(pl.when(excluded)
                             .then(pl.lit(_UNRECOVERABLE))
                             .otherwise(pl.lit(None, dtype=pl.Utf8))
                             .alias("filter_reason"))
             .select(_V6_REPERTOIRE_COLUMNS)
             .sink_parquet(tmp))
        tmp.replace(f)


def _split_repertoire_meta(db_dir: Path) -> None:
    """One table per locus -> one locus-invariant table + one counts table per locus.

    The invariant half was copied into every locus file, so the split is a dedupe on
    `repertoire_id` across them. A v5 table may be missing a column outright — a tree that
    came up from v3 never had `source_files` — so each half is completed with typed nulls
    rather than assumed. Sources are deleted last, which makes the step idempotent: a re-run
    finds no per-locus tables and leaves what it already wrote.
    """
    meta_dir = db_dir / _META_REPERTOIRE
    sources = [(locus, meta_dir / f"{locus}.parquet") for locus in _V5_META_LOCI]
    sources = [(locus, path) for locus, path in sources if path.exists()]
    if not sources:
        return

    invariant = [_complete(pl.read_parquet(p), _V6_META_COLUMNS) for _, p in sources]
    (pl.concat(invariant)
       .unique(subset=["repertoire_id"], keep="first")
       .sort("repertoire_id")
       .write_parquet(meta_dir / "repertoire.parquet"))

    for locus, path in sources:
        if locus == _UNASSIGNED:
            continue          # v6 counts cover the real loci; the QC bucket has no counts table
        out = meta_dir / f"locus={locus}" / "counts.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        _complete(pl.read_parquet(path), _V6_COUNTS_COLUMNS).write_parquet(out)

    for _, path in sources:
        path.unlink()


def _complete(df: pl.DataFrame, columns: dict) -> pl.DataFrame:
    """`df` reduced to exactly `columns`, in order, with anything absent added as typed nulls."""
    missing = [pl.lit(None, dtype=dtype).alias(name)
               for name, dtype in columns.items() if name not in df.columns]
    return df.with_columns(missing).select(list(columns)).cast(columns)


def _rename_files(db_dir: Path) -> None:
    """Pure renames. A destination that already exists means this step ran before."""
    for src, dst in _RENAMES.items():
        source, target = db_dir / src, db_dir / dst
        if source.exists() and not target.exists():
            shutil.move(str(source), str(target))


def _clear_operations(db_dir: Path) -> None:
    """Results are regenerable, and their records describe an output set this bump changed."""
    ops = db_dir / "operations"
    if not ops.is_dir() or not any(ops.iterdir()):
        return
    shutil.rmtree(ops)
    ops.mkdir()
    warnings.warn("v6 cleared operations/: operation records key on an output set that changed "
                  "in this version. Re-run the operations you need.", stacklevel=2)


def _drop_present_loci(db_dir: Path) -> None:
    """`present_loci` is now the shard listing, so the copy in the manifest is deleted.

    Raw JSON read-modify-write: every other field must survive untouched, including ones this
    version has never heard of.
    """
    path = db_dir / "meta/manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    if manifest.pop("present_loci", None) is None:
        return
    path.write_text(json.dumps(manifest, indent=2))
