import polars as pl

GENERATION_META = pl.Schema({
    "dataset_name": pl.Utf8,
    "created_on": pl.Date,
    "source": pl.Utf8,
    "reader" : pl.Utf8,
    "repertoire_mapper": pl.Utf8,
    "patient_mapper": pl.Utf8,
}) # ndjson

REPERTOIRE = pl.Schema({
    "repertoire_id": pl.Utf8,
    "junction": pl.Utf8,
    "v_call": pl.Utf8,
    "junction_aa": pl.Utf8,
    "j_call": pl.Utf8,
    "duplicate_count": pl.Int64,
    "filter_pass": pl.Boolean,
}) # parquet

REPERTOIRE_META = pl.Schema({
    "repertoire_id": pl.Utf8,
    "source_files" : pl.List(pl.Utf8),
    "patient_id": pl.Utf8,
    "n_clonotypes": pl.Int64,
    "n_filtered_clonotypes":pl.Int64,
    "total_duplicates": pl.Int64
}) # parquet

PATIENT_META = pl.Schema({
    "patient_id": pl.Utf8,
    "patient_repertoires": pl.List(pl.Utf8),
    "n_repertoires": pl.Int64,
}) # parquet

PUBLICATION_META = pl.Schema({
    "publication_id": pl.Utf8,
}) # ndjson

# Operation provenance is no longer a global polars-schema table. Each op writes a
# self-describing `operations/<op>/operation.json` (the `OperationRecord` dataclass);
# see tcr_io/operations/base.py.