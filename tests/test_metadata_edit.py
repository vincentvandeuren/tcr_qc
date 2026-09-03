"""Metadata write path: `ds.metadata.set_*` and the pure
the `MetadataWriter` API end to end. See
docs/superpowers/specs/2026-07-15-metadata-write-api-design.md.

The four pure frame helpers had their own unit tests here; they were folded into
`tcr_io.metadata` as private functions and those tests deleted. Every one of them is exercised
through the writer below — a merge that adds a column, a duplicate key that raises, a colliding
column that warns — which is where their behaviour actually matters."""
import json
import tempfile
import warnings
from pathlib import Path

import polars as pl
import pytest

from tcr_io import Dataset
from tcr_io.structure import (
    ARTIFACTS, Store, Manifest,
    REPERTOIRE_FILE, REPERTOIRE_META, REPERTOIRE_COUNTS, PATIENT_META, PATIENT_EXTRA,
    PUBLICATION_IDS, GENERATION_META, MANIFEST, HLA,
)
import tcr_io.structure.schema as S


# ── fixture: minimal 2-patient / 2-repertoire dataset ────────────────────────────

def _make_dataset() -> Path:
    d = Path(tempfile.mkdtemp()) / "ds"
    Store(d).create_tree(ARTIFACTS)

    reps = ["rep_a", "rep_b"]
    patients = ["p1", "p2"]
    for rid in reps:
        f = d / REPERTOIRE_FILE.template.format(repertoire_id=rid, locus="TRB")
        f.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "repertoire_id": [rid, rid],
            "junction": ["TGT", "TGC"],
            "v_call": ["TRBV2*01", "TRBV7-2*01"],
            "junction_aa": ["CASSF", "CASSY"],
            "j_call": ["TRBJ2-1*01", "TRBJ2-1*01"],
            "duplicate_count": [10, 5],
            # v6: null reason == the row passed (`passed <=> filter_reason.is_null()`)
            "filter_reason": [None, None],
        }).with_columns(clonotype_id=pl.int_range(pl.len(), dtype=pl.UInt32)).cast(S.REPERTOIRE).write_parquet(f)

    # v6 splits this by grain: what holds of a repertoire regardless of locus, and what only
    # holds within one. Two tables, two writes.
    pl.DataFrame({
        "repertoire_id": reps, "source_files": [["a"], ["b"]], "patient_id": patients,
    }).cast(S.REPERTOIRE_META).write_parquet(d / REPERTOIRE_META.template)
    counts = d / REPERTOIRE_COUNTS.template.format(locus="TRB")
    counts.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "repertoire_id": reps, "n_clonotypes": [2, 2], "n_filtered_clonotypes": [0, 0],
        "total_duplicates": [15, 15],
    }).cast(S.REPERTOIRE_COUNTS).write_parquet(counts)
    pl.DataFrame({
        "patient_id": patients, "patient_repertoires": [["rep_a"], ["rep_b"]], "n_repertoires": [1, 1],
    }).cast(S.PATIENT_META).write_parquet(d / PATIENT_META.template)
    pl.DataFrame({"publication_id": ["pub1"]}, schema=S.PUBLICATION_META).write_ndjson(d / PUBLICATION_IDS.template)
    (d / GENERATION_META.template).write_text(json.dumps(
        {"dataset_name": "t", "created_on": "2026-01-01", "source": "x", "tcrio_version": "t",
         "reader": "r", "repertoire_mapper": "m", "patient_mapper": "m", "filters": {}}))
    Manifest.current().write(Store(d))
    return d


def _ds() -> Dataset:
    warnings.simplefilter("ignore")
    return Dataset(_make_dataset())


def test_set_patient_meta_replace_then_read_back():
    ds = _ds()
    ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["M", "F"]}), mode="replace")
    full = ds.full_patient_meta.sort("patient_id")
    assert "sex" in full.columns
    assert full.get_column("sex").to_list() == ["M", "F"]


def test_set_patient_meta_merge_adds_new_column_keeps_existing():
    ds = _ds()
    ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["M", "F"]}), mode="merge")
    ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1", "p2"], "age": [30, 40]}), mode="merge")
    full = ds.full_patient_meta.sort("patient_id")
    assert {"sex", "age"} <= set(full.columns)
    assert full.get_column("sex").to_list() == ["M", "F"]
    assert full.get_column("age").to_list() == [30, 40]


def test_set_patient_meta_merge_adds_new_row():
    ds = _ds()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1"], "sex": ["M"]}), mode="merge")   # p2 missing (warns)
        ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p2"], "sex_2": ["F"]}), mode="merge")  # p2 now added
    extra = pl.read_parquet(ds.db_dir / PATIENT_EXTRA.template).sort("patient_id")
    assert extra.get_column("patient_id").to_list() == ["p1", "p2"]


def test_merge_duplicate_column_warns_and_is_skipped():
    ds = _ds()
    ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["M", "F"]}), mode="merge")
    with pytest.warns(UserWarning, match="already exist"):
        ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["X", "X"]}), mode="merge")
    # original values are untouched (merge never overwrites an existing column)
    assert ds.full_patient_meta.sort("patient_id").get_column("sex").to_list() == ["M", "F"]


def test_column_collision_with_base_warns():
    ds = _ds()
    # 'n_repertoires' is a base patient_meta column -> collision
    with pytest.warns(UserWarning, match="already exist"):
        ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1", "p2"], "n_repertoires": [9, 9]}), mode="replace")
    extra = pl.read_parquet(ds.db_dir / PATIENT_EXTRA.template)
    assert "n_repertoires" not in extra.columns


def test_duplicate_keys_raises():
    ds = _ds()
    with pytest.raises(ValueError, match="duplicate"):
        ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1", "p1"], "sex": ["M", "F"]}))


def test_unknown_key_warns_but_writes():
    ds = _ds()
    with pytest.warns(UserWarning, match="not present in the dataset"):
        ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1", "p2", "p999"], "sex": ["M", "F", "?"]}), mode="replace")
    # unknown row is written but never surfaces in the left-join
    assert "p999" not in ds.full_patient_meta.get_column("patient_id").to_list()


def test_missing_key_warns():
    ds = _ds()
    with pytest.warns(UserWarning, match="no row in the input"):
        ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1"], "sex": ["M"]}), mode="replace")


def test_bad_mode_raises():
    ds = _ds()
    with pytest.raises(ValueError, match="mode must be"):
        ds.metadata.set_patient(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["M", "F"]}), mode="upsert")


def test_set_repertoire_meta_read_back():
    ds = _ds()
    ds.metadata.set_repertoire(
        pl.DataFrame({"repertoire_id": ["rep_a", "rep_b"], "tissue": ["blood", "tumor"]}), mode="replace")
    full = ds.select_locus("TRB").full_repertoire_meta.sort("repertoire_id")
    assert full.get_column("tissue").to_list() == ["blood", "tumor"]


# ── integration: HLA (replace-only, fixed schema) ────────────────────────────────

def _hla_df(patients):
    n = len(patients)
    return pl.DataFrame({
        "patient_id": patients,
        "A": [["0201", "0101"]] * n, "B": [["4403"]] * n, "C": [["0702"]] * n,
        "DRB1": [["1501"]] * n, "DPA1": [["0103"]] * n, "DPB1": [["0401"]] * n,
        "DQA1": [["0102"]] * n, "DQB1": [["0602"]] * n,
    })


def test_set_hla_read_back():
    ds = _ds()
    ds.metadata.set_hla(_hla_df(["p1", "p2"]))
    full = ds.full_patient_meta.sort("patient_id")
    assert {"A", "B", "DRB1", "DQB1"} <= set(full.columns)
    assert full.filter(pl.col("patient_id") == "p1").get_column("A").to_list()[0] == ["0201", "0101"]


def test_set_hla_partial_loci_filled():
    ds = _ds()
    partial = pl.DataFrame({"patient_id": ["p1", "p2"], "A": [["0201"], ["0101"]]})   # only locus A
    ds.metadata.set_hla(partial)
    hla = pl.read_parquet(ds.db_dir / HLA.template)
    assert set(hla.columns) == set(S.HLA_META.keys())
    assert hla.get_column("B").to_list() == [None, None]                              # absent loci -> null list


def test_set_hla_extra_column_warns_and_dropped():
    ds = _ds()
    df = _hla_df(["p1", "p2"]).with_columns(pl.lit("note").alias("comment"))
    with pytest.warns(UserWarning, match="not part of the HLA schema"):
        ds.metadata.set_hla(df)
    assert "comment" not in pl.read_parquet(ds.db_dir / HLA.template).columns


def test_set_hla_bad_dtype_raises():
    ds = _ds()
    # A as a plain string instead of List(str): must raise, not silently wrap into ["0201"]
    bad = pl.DataFrame({"patient_id": ["p1", "p2"], "A": ["0201", "0101"]})
    with pytest.raises(ValueError, match="must be List"):
        ds.metadata.set_hla(bad)


def test_set_hla_missing_key_warns():
    ds = _ds()
    with pytest.warns(UserWarning, match="no row in the input"):
        ds.metadata.set_hla(_hla_df(["p1"]))    # p2 untyped -> missing warning (expected for HLA)


def test_set_hla_is_replace_only():
    ds = _ds()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.metadata.set_hla(_hla_df(["p1", "p2"]))
        ds.metadata.set_hla(pl.DataFrame({"patient_id": ["p1"], "A": [["9999"]]}))   # replaces whole table
    hla = pl.read_parquet(ds.db_dir / HLA.template)
    assert hla.get_column("patient_id").to_list() == ["p1"]                 # p2 gone (full replace)
    assert hla.get_column("B").to_list() == [None]


def test_set_publication_writes_the_table_the_reader_already_joined():
    """`full_publication_meta` has always left-joined `PUBLICATION_EXTRA`; until MetadataWriter
    there was no way to put anything in it."""
    ds = Dataset(_make_dataset())
    ds.metadata.set_publication(pl.DataFrame({"publication_id": ["pub1"], "cohort": ["c1"]}))
    assert ds.full_publication_meta.row(by_predicate=pl.col("publication_id") == "pub1",
                                        named=True)["cohort"] == "c1"


def test_set_publication_rejects_an_unknown_id():
    ds = Dataset(_make_dataset())
    with pytest.warns(UserWarning, match="not present in the dataset"):
        ds.metadata.set_publication(pl.DataFrame({"publication_id": ["nope"], "cohort": ["c"]}))


# ── fetch_publications: the one writer that derives its own rows ─────────────────

def _study(**kw):
    from tcr_io.publications import StudyMetadata
    return StudyMetadata(**kw)


def test_fetch_publications_writes_what_the_client_returned(monkeypatch):
    ds = Dataset(_make_dataset())
    monkeypatch.setattr("tcr_io.metadata.fetch_publication",
                        lambda pid: _study(title="T", authors=["A B"], year=2017,
                                           source_type="pmc", doi="10.1/x"))
    ds.metadata.fetch_publications()
    row = ds.full_publication_meta.row(by_predicate=pl.col("publication_id") == "pub1", named=True)
    assert (row["title"], row["year"], row["authors"], row["doi"]) == ("T", 2017, ["A B"], "10.1/x")


def test_fetch_publications_refreshes_its_own_columns_and_keeps_the_users(monkeypatch):
    """The rule from both sides: fetched columns belong to the fetcher, everything else
    to `set_publication`."""
    ds = Dataset(_make_dataset())
    monkeypatch.setattr("tcr_io.metadata.fetch_publication",
                        lambda pid: _study(title="stale", source_type="doi"))
    ds.metadata.fetch_publications()
    ds.metadata.set_publication(pl.DataFrame({"publication_id": ["pub1"], "cohort": ["c1"]}))

    monkeypatch.setattr("tcr_io.metadata.fetch_publication",
                        lambda pid: _study(title="fresh", source_type="doi"))
    ds.metadata.fetch_publications()

    row = ds.full_publication_meta.row(by_predicate=pl.col("publication_id") == "pub1", named=True)
    assert row["title"] == "fresh"      # replaced, not skipped as a collision
    assert row["cohort"] == "c1"        # the user's column survived the refresh


def test_fetch_publications_records_an_unusable_identifier_offline():
    """No network: 'pub1' matches no source shape, so it comes back as an error row rather
    than as no row at all."""
    ds = Dataset(_make_dataset())
    ds.metadata.fetch_publications()
    row = ds.full_publication_meta.row(by_predicate=pl.col("publication_id") == "pub1", named=True)
    assert row["source_type"] == "unknown" and row["error"] == "unrecognized identifier"
    assert row["title"] is None
