import polars as pl

# The ingest record (`meta/generation.json`) is a JSON object, not a table: one dataset, one
# row, and a nested `filters` value that a polars schema can only express as a struct column.
# It answers "how was this dataset made" — source, mappers, reader, library version and the
# filter set applied per locus — and it is the only record of the cleaning that ingestion does
# in place. Written by `DatasetIngester`, read by `ds.generation_meta`. No schema here: it is a
# self-describing dict, like `OperationRecord`.

REPERTOIRE = pl.Schema({
    "clonotype_id": pl.UInt32,   # per-(repertoire, locus) row number (assigned at ingest)
    "repertoire_id": pl.Utf8,
    "junction": pl.Utf8,
    "v_call": pl.Utf8,
    "junction_aa": pl.Utf8,
    "j_call": pl.Utf8,
    "duplicate_count": pl.Int64,
    # Null when the row passed ingest filtering, otherwise the name of the FIRST filter that
    # rejected it. The invariant is `passed <=> filter_reason.is_null()`, so this replaces the
    # old boolean rather than sitting beside it — and the reason a row was dropped stops being
    # unrecoverable, which is what let the filtering report become a group-by instead of a
    # rebuild-from-provenance.
    "filter_reason": pl.Utf8,
}) # parquet

# Single-cell only: the clonotype<->cell mapping, long form (one row per cell membership).
# clonotype_id is scoped by (repertoire_id, locus) — the same grain as the hive partition — so a
# cell's paired chains (e.g. TRA + TRB) are distinct rows.
CLONE_TO_CELL = pl.Schema({
    "repertoire_id": pl.Utf8,
    "locus": pl.Utf8,
    "clonotype_id": pl.UInt32,
    "cell_id": pl.Utf8,
}) # parquet

# One row per repertoire, locus-invariant. Owned by the dataset, not by a locus.
REPERTOIRE_META = pl.Schema({
    "repertoire_id": pl.Utf8,
    "source_files" : pl.List(pl.Utf8),
    "patient_id": pl.Utf8,
}) # parquet

# One row per repertoire *within one locus*, stored under that locus's partition. A repertoire
# appears only in the loci it actually produced.
REPERTOIRE_COUNTS = pl.Schema({
    "repertoire_id": pl.Utf8,
    "n_clonotypes": pl.Int64,           # rows that passed filtering
    "n_filtered_clonotypes": pl.Int64,  # rows that were rejected
    "total_duplicates": pl.Int64,
}) # parquet

PATIENT_META = pl.Schema({
    "patient_id": pl.Utf8,
    "patient_repertoires": pl.List(pl.Utf8),
    "n_repertoires": pl.Int64,
}) # parquet

PUBLICATION_META = pl.Schema({
    "publication_id": pl.Utf8,
}) # ndjson

# Publication metadata fetched from the web by `MetadataWriter.fetch_publications`, one row per
# publication_id, joined into `full_publication_meta`. Written into the free-form publication
# side table, so these are the column names the writer owns: a refetch replaces them and leaves
# the user's own columns alone. `error` set means the fetch failed and the rest is null.
PUBLICATION_FETCHED = pl.Schema({
    "publication_id": pl.Utf8,
    "title": pl.Utf8,
    "abstract": pl.Utf8,
    "authors": pl.List(pl.Utf8),
    "year": pl.Int64,
    "source_type": pl.Utf8,      # doi / pmc / bioproject / adaptive / unknown
    "doi": pl.Utf8,
    "pmid": pl.Utf8,
    "journal": pl.Utf8,
    "error": pl.Utf8,
}) # parquet

# Ground-truth ("known") HLA typing, one row per patient. Each locus column holds the
# patient's alleles as bare 4-digit strings (e.g. ["0201", "2902"]); an untyped locus is a
# null/empty list. Input must already be parsed to this canonical notation — the write path
# does no parsing, it only validates the container/dtype. Joined into `full_patient_meta`.
HLA_META = pl.Schema({
    "patient_id": pl.Utf8,
    "A":    pl.List(pl.Utf8),
    "B":    pl.List(pl.Utf8),
    "C":    pl.List(pl.Utf8),
    "DRB1": pl.List(pl.Utf8),
    "DPA1": pl.List(pl.Utf8),
    "DPB1": pl.List(pl.Utf8),
    "DQA1": pl.List(pl.Utf8),
    "DQB1": pl.List(pl.Utf8),
}) # parquet

# Operation provenance is no longer a global polars-schema table. Each op writes a
# self-describing `operations/<op>/operation.json` (the `OperationRecord` dataclass);
# see tcr_io/operations/base.py.