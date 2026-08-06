"""Metadata write path: `set_patient_meta` / `set_repertoire_meta` / `set_hla` and the pure
`structure.meta_edit` helpers. See docs/superpowers/specs/2026-07-15-metadata-write-api-design.md."""
import tempfile
import warnings
from pathlib import Path

import polars as pl
import pytest

from tcr_io import TcrDataset
from tcr_io.structure import (
    Layout, REQUIRED_DIRS, GENERATED_DIRS, Manifest,
    repertoire_locus_relpath, repertoire_meta_relpath, meta_edit,
)
import tcr_io.structure.schema as S


# ── fixture: minimal 2-patient / 2-repertoire dataset ────────────────────────────

def _make_dataset() -> Path:
    d = Path(tempfile.mkdtemp()) / "ds"
    for sub in REQUIRED_DIRS + GENERATED_DIRS:
        (d / sub).mkdir(parents=True, exist_ok=True)

    reps = ["rep_a", "rep_b"]
    patients = ["p1", "p2"]
    for rid in reps:
        f = d / repertoire_locus_relpath(rid, "TRB")
        f.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "repertoire_id": [rid, rid],
            "junction": ["TGT", "TGC"],
            "v_call": ["TRBV2*01", "TRBV7-2*01"],
            "junction_aa": ["CASSF", "CASSY"],
            "j_call": ["TRBJ2-1*01", "TRBJ2-1*01"],
            "duplicate_count": [10, 5],
            "filter_pass": [True, True],
        }).with_columns(clonotype_id=pl.int_range(pl.len(), dtype=pl.UInt32)).cast(S.REPERTOIRE).write_parquet(f)

    pl.DataFrame({
        "repertoire_id": reps, "source_files": [["a"], ["b"]], "patient_id": patients,
        "n_clonotypes": [2, 2], "n_filtered_clonotypes": [2, 2], "total_duplicates": [15, 15],
    }).cast(S.REPERTOIRE_META).write_parquet(d / repertoire_meta_relpath("TRB"))
    pl.DataFrame({
        "patient_id": patients, "patient_repertoires": [["rep_a"], ["rep_b"]], "n_repertoires": [1, 1],
    }).cast(S.PATIENT_META).write_parquet(d / Layout.patient_meta.path)
    pl.DataFrame({"publication_id": []}, schema=S.PUBLICATION_META).write_ndjson(d / Layout.publication_ids.path)
    pl.DataFrame(
        {"dataset_name": ["t"], "created_on": [None], "source": ["x"],
         "reader": ["r"], "repertoire_mapper": ["m"], "patient_mapper": ["m"]}
    ).cast(S.GENERATION_META).write_ndjson(d / Layout.generation_meta.path)
    Manifest.current(present_loci=["TRB"]).write(d / Layout.manifest.path)
    return d


def _ds() -> TcrDataset:
    warnings.simplefilter("ignore")
    return TcrDataset(_make_dataset())


# ── unit: pure helpers ───────────────────────────────────────────────────────────

def test_duplicate_keys():
    assert meta_edit.duplicate_keys(pl.DataFrame({"k": ["a", "b", "a"]}), "k") == ["a"]
    assert meta_edit.duplicate_keys(pl.DataFrame({"k": ["a", "b"]}), "k") == []


def test_check_keys():
    unknown, missing = meta_edit.check_keys(["p1", "p3"], ["p1", "p2"])
    assert unknown == ["p3"]
    assert missing == ["p2"]
    assert meta_edit.check_keys(["p1", "p2"], ["p1", "p2"]) == ([], [])


def test_reserved_column_collisions():
    cols = ["patient_id", "age", "disease"]
    assert meta_edit.reserved_column_collisions(cols, ["patient_id", "disease"], "patient_id") == ["disease"]
    # the key itself never counts as a collision
    assert meta_edit.reserved_column_collisions(cols, ["patient_id"], "patient_id") == []


def test_merge_frames_adds_columns_and_rows():
    old = pl.DataFrame({"k": ["p1", "p2"], "disease": ["flu", "cov"]})
    new = pl.DataFrame({"k": ["p1", "p3"], "age": [30, 40]})
    out = meta_edit.merge_frames(old, new, "k").sort("k")

    assert set(out.columns) == {"k", "disease", "age"}
    assert out.get_column("k").to_list() == ["p1", "p2", "p3"]
    row = {r["k"]: r for r in out.to_dicts()}
    assert row["p1"]["disease"] == "flu" and row["p1"]["age"] == 30
    assert row["p2"]["disease"] == "cov" and row["p2"]["age"] is None      # existing row, no new value
    assert row["p3"]["disease"] is None and row["p3"]["age"] == 40         # new row


# ── integration: free-form setters ───────────────────────────────────────────────

def test_set_patient_meta_replace_then_read_back():
    ds = _ds()
    ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["M", "F"]}), mode="replace")
    full = ds.full_patient_meta.sort("patient_id")
    assert "sex" in full.columns
    assert full.get_column("sex").to_list() == ["M", "F"]


def test_set_patient_meta_merge_adds_new_column_keeps_existing():
    ds = _ds()
    ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["M", "F"]}), mode="merge")
    ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p2"], "age": [30, 40]}), mode="merge")
    full = ds.full_patient_meta.sort("patient_id")
    assert {"sex", "age"} <= set(full.columns)
    assert full.get_column("sex").to_list() == ["M", "F"]
    assert full.get_column("age").to_list() == [30, 40]


def test_set_patient_meta_merge_adds_new_row():
    ds = _ds()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1"], "sex": ["M"]}), mode="merge")   # p2 missing (warns)
        ds.set_patient_meta(pl.DataFrame({"patient_id": ["p2"], "sex_2": ["F"]}), mode="merge")  # p2 now added
    extra = pl.read_parquet(ds.db_dir / Layout.patient_meta_extra.path).sort("patient_id")
    assert extra.get_column("patient_id").to_list() == ["p1", "p2"]


def test_merge_duplicate_column_warns_and_is_skipped():
    ds = _ds()
    ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["M", "F"]}), mode="merge")
    with pytest.warns(UserWarning, match="already exist"):
        ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["X", "X"]}), mode="merge")
    # original values are untouched (merge never overwrites an existing column)
    assert ds.full_patient_meta.sort("patient_id").get_column("sex").to_list() == ["M", "F"]


def test_column_collision_with_base_warns():
    ds = _ds()
    # 'n_repertoires' is a base patient_meta column -> collision
    with pytest.warns(UserWarning, match="already exist"):
        ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p2"], "n_repertoires": [9, 9]}), mode="replace")
    extra = pl.read_parquet(ds.db_dir / Layout.patient_meta_extra.path)
    assert "n_repertoires" not in extra.columns


def test_duplicate_keys_raises():
    ds = _ds()
    with pytest.raises(ValueError, match="duplicate"):
        ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p1"], "sex": ["M", "F"]}))


def test_unknown_key_warns_but_writes():
    ds = _ds()
    with pytest.warns(UserWarning, match="not present in the dataset"):
        ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p2", "p999"], "sex": ["M", "F", "?"]}), mode="replace")
    # unknown row is written but never surfaces in the left-join
    assert "p999" not in ds.full_patient_meta.get_column("patient_id").to_list()


def test_missing_key_warns():
    ds = _ds()
    with pytest.warns(UserWarning, match="no row in the input"):
        ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1"], "sex": ["M"]}), mode="replace")


def test_bad_mode_raises():
    ds = _ds()
    with pytest.raises(ValueError, match="mode must be"):
        ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p2"], "sex": ["M", "F"]}), mode="upsert")


def test_set_repertoire_meta_read_back():
    ds = _ds()
    ds.set_repertoire_meta(
        pl.DataFrame({"repertoire_id": ["rep_a", "rep_b"], "tissue": ["blood", "tumor"]}), mode="replace")
    full = ds.full_repertoire_meta("TRB").sort("repertoire_id")
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
    ds.set_hla(_hla_df(["p1", "p2"]))
    full = ds.full_patient_meta.sort("patient_id")
    assert {"A", "B", "DRB1", "DQB1"} <= set(full.columns)
    assert full.filter(pl.col("patient_id") == "p1").get_column("A").to_list()[0] == ["0201", "0101"]


def test_set_hla_partial_loci_filled():
    ds = _ds()
    partial = pl.DataFrame({"patient_id": ["p1", "p2"], "A": [["0201"], ["0101"]]})   # only locus A
    ds.set_hla(partial)
    hla = pl.read_parquet(ds.db_dir / Layout.hla.path)
    assert set(hla.columns) == set(S.HLA_META.keys())
    assert hla.get_column("B").to_list() == [None, None]                              # absent loci -> null list


def test_set_hla_extra_column_warns_and_dropped():
    ds = _ds()
    df = _hla_df(["p1", "p2"]).with_columns(pl.lit("note").alias("comment"))
    with pytest.warns(UserWarning, match="not part of the HLA schema"):
        ds.set_hla(df)
    assert "comment" not in pl.read_parquet(ds.db_dir / Layout.hla.path).columns


def test_set_hla_bad_dtype_raises():
    ds = _ds()
    # A as a plain string instead of List(str): must raise, not silently wrap into ["0201"]
    bad = pl.DataFrame({"patient_id": ["p1", "p2"], "A": ["0201", "0101"]})
    with pytest.raises(ValueError, match="must be List"):
        ds.set_hla(bad)


def test_set_hla_missing_key_warns():
    ds = _ds()
    with pytest.warns(UserWarning, match="no row in the input"):
        ds.set_hla(_hla_df(["p1"]))    # p2 untyped -> missing warning (expected for HLA)


def test_set_hla_is_replace_only():
    ds = _ds()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.set_hla(_hla_df(["p1", "p2"]))
        ds.set_hla(pl.DataFrame({"patient_id": ["p1"], "A": [["9999"]]}))   # replaces whole table
    hla = pl.read_parquet(ds.db_dir / Layout.hla.path)
    assert hla.get_column("patient_id").to_list() == ["p1"]                 # p2 gone (full replace)
    assert hla.get_column("B").to_list() == [None]
