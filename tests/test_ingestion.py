"""Ingestion behaviours that don't need the full reader/mapper pipeline."""
import json
import tempfile
from pathlib import Path

import polars as pl

from tcr_io.ingestion import DatasetIngester
from tcr_io.readers import AirrReader
from tcr_io.mappers import FileNameMapper, DictMapper
from tcr_io.expressions import assign_locus, extract_locus, UNASSIGNED
from tcr_io.filters import FilterSet, PerLocusFilterSet, NullVFilter, NullJFilter
from tcr_io.structure import DATASET_VERSION, Manifest, Store, OPERATIONS_DIR, PROCESSED_DIR, MANIFEST


def _seed_db() -> Path:
    """A minimal existing dataset with a processed repertoire and one operation result."""
    root = Path(tempfile.mkdtemp())
    db = root / "ds"
    (db / PROCESSED_DIR.template).mkdir(parents=True)
    (db / PROCESSED_DIR.template / "rep_a.parquet").write_text("x")
    opdir = db / OPERATIONS_DIR / "diversity_report"
    (opdir / "TRB").mkdir(parents=True)
    (opdir / "operation.json").write_text(json.dumps({"status": "success"}))
    (opdir / "TRB" / "diversity_summary.parquet").write_text("y")
    return root


def test_reingest_clears_stale_operation_outputs():
    # Re-ingesting the source makes every derived op output stale, so _delete_existing_data
    # must clear operations/ (not just processed_repertoires) — else run_operation would skip
    # them as "already done" and serve results computed from the old inputs.
    root = _seed_db()
    ing = DatasetIngester(db_dir=root, db_name="ds", repertoire_mapper=None, patient_mapper=None)
    ing._delete_existing_data()
    assert not (ing.db_dir / PROCESSED_DIR.template).exists()
    assert not (ing.db_dir / OPERATIONS_DIR).exists()


def test_delete_existing_data_ok_without_operations_dir():
    # Older datasets may have no operations/ yet — clearing must not raise.
    root = _seed_db()
    import shutil
    shutil.rmtree(root / "ds" / OPERATIONS_DIR)
    ing = DatasetIngester(db_dir=root, db_name="ds", repertoire_mapper=None, patient_mapper=None)
    ing._delete_existing_data()          # must not raise
    assert not (ing.db_dir / PROCESSED_DIR.template).exists()


def _mixed_locus_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "v_call": ["TRBV2*01", "TRAV1-1*01", "XXYV9*01", "TRBV5*01", None],
        "j_call": ["TRBJ2-1*01", "TRAJ1*01", "XXYJ1*01", "TRAJ2*01", "TRBJ1-1*01"],
    })


def test_assign_locus_maps_known_and_unassigned():
    out = _mixed_locus_frame().with_columns(assign_locus())["locus"].to_list()
    # TRB, TRA are known; "XX" prefix is unknown; TRB/TRA prefix mismatch; null v_call -> all _unassigned
    assert out == ["TRB", "TRA", UNASSIGNED, UNASSIGNED, UNASSIGNED]


def test_extract_locus_trav_dv_disambiguated_by_j():
    # TRAV/DV genes are shared between TRA and TRD; the J gene decides. Their "TRA" prefix must
    # NOT force a TRA/TRD mismatch to null. Any non-TRA/TRD J prefix -> undefined -> null.
    frame = pl.DataFrame({
        "v_call": ["TRAV14/DV4*01", "TRAV14/DV4*01", "TRAV14/DV4*01", "TRAV38-2/DV8*01"],
        "j_call": ["TRAJ1*01",      "TRDJ1*01",      "TRBJ2-1*01",    "TRDJ3*01"],
    })
    out = frame.with_columns(extract_locus())["locus"].to_list()
    assert out == ["TRA", "TRD", None, "TRD"]


def test_filter_applied_after_locus_assignment():
    """Phase 2: quality filters run after assign_locus. Assigned-locus rows get the reason
    their own filter set gave them; _unassigned rows (null/mismatch/unknown-locus) carry the
    ASSIGNMENT reason instead — they were never tested against a quality filter."""
    d = Path(tempfile.mkdtemp())
    src = d / "data"; src.mkdir()
    pl.DataFrame({
        # a 39nt junction survives CDR3 trimming; TGTGCC is too short and would trim to null
        "junction":    ["TGTGCCAGCAGCTTAGGGTATGAACAGTACTTC"] * 4 + ["TGTGCC"],
        "junction_aa": ["CASSLGYEQYF", "CASSLGYEQYF", "CASSLGYEQYF", "CASSLGYEQYF", "CASSLGYEQYF"],
        "v_call":      ["TRBV2*01",    "TRBV999*01",  "TRBV2*01",    "TRBV2*01",    None],
        "j_call":      ["TRBJ2-1*01",  "TRBJ2-1*01",  "TRBJ2-1*01",  "TRAJ2*01",    "TRBJ1-1*01"],
        "duplicate_count": [5, 3, 2, 4, 1],
    }).write_csv(src / "sampleA.tsv", separator="\t")

    ds = DatasetIngester(
        db_dir=d, db_name="db",
        repertoire_mapper=FileNameMapper(),
        patient_mapper=DictMapper({"sampleA.tsv": "p1"}),
        reader=AirrReader(),
    ).run(src)

    assert sorted(ds.present_loci) == ["TRB"]

    # v6 is locus-first: `locus={L}/{id}.parquet`, not `{id}/locus={L}/{id}.parquet`
    base = d / "db" / "processed_repertoires"
    trb = pl.read_parquet(base / "locus=TRB" / "sampleA.tsv.parquet")
    # null reason == passed. The invalid-gene row names the filter that rejected it, which the
    # old boolean could not: `filter_reason` is what made the filtering report a group-by.
    assert trb.filter(pl.col("v_call") == "TRBV2*01")["filter_reason"].to_list() == [None]
    assert trb.filter(pl.col("v_call") == "TRBV999*01")["filter_reason"].to_list() == ["invalid_v_call"]

    una = pl.read_parquet(base / "locus=_unassigned" / "sampleA.tsv.parquet")
    assert una.height == 2                                   # TRB/TRA mismatch + null v_call
    assert una["filter_reason"].is_null().sum() == 0         # _unassigned is always excluded
    assert set(una["filter_reason"]) == {"incompatible_locus", "null_v"}


def test_ingest_records_filter_provenance():
    """The filter set(s) used at ingest are recorded in the ingest record.

    Ingestion cleans in place and `filter_reason` names only the filters that FIRED, so this
    is the only record of what was applied."""
    d = Path(tempfile.mkdtemp())
    src = d / "data"; src.mkdir()
    pl.DataFrame({
        "junction": ["TGTGCCAGCAGCTTAGGGTATGAACAGTACTTC"],
        "junction_aa": ["CASSLGYEQYF"], "v_call": ["TRBV2*01"], "j_call": ["TRBJ2-1*01"],
        "duplicate_count": [5],
    }).write_csv(src / "sampleA.tsv", separator="\t")

    # explicit chains->filters mapping: TRB gets a custom set, TRA gets the default preset
    override = FilterSet([NullVFilter(), NullJFilter()])
    ds = DatasetIngester(
        db_dir=d, db_name="db",
        repertoire_mapper=FileNameMapper(),
        patient_mapper=DictMapper({"sampleA.tsv": "p1"}),
        reader=AirrReader(),
        filter_set=PerLocusFilterSet({("TRB",): override, ("TRA",): FilterSet.named("default_trb")}),
    ).run(src)

    filters = ds.generation_meta["filters"]
    assert filters["TRB"]["filters"] == ["null_v", "null_j"]
    assert filters["TRA"]["preset"] == "default_trb"
    assert "imgt_functional" in filters["TRA"]["filters"]
    assert "*" not in filters                       # no catch-all default anymore
    # A preset name re-runs; an ad-hoc set has none, and its expansion is all there is.
    assert filters["TRB"]["preset"] is None
    assert FilterSet.named(filters["TRA"]["preset"]).names == filters["TRA"]["filters"]
    assert Manifest.read(Store(d / "db")).version == DATASET_VERSION   # manifest holds only this


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok:", _name)
