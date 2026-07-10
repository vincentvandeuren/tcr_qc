"""Ingestion behaviours that don't need the full reader/mapper pipeline."""
import json
import tempfile
from pathlib import Path

import polars as pl

from tcr_io.ingestion import DatasetIngester
from tcr_io.expressions import partition_by_locus
from tcr_io.structure import Layout


def _seed_db() -> Path:
    """A minimal existing dataset with a processed repertoire and one operation result."""
    root = Path(tempfile.mkdtemp())
    db = root / "ds"
    (db / Layout.processed_dir.path).mkdir(parents=True)
    (db / Layout.processed_dir.path / "rep_a.parquet").write_text("x")
    opdir = db / Layout.operations_dir.path / "diversity_report"
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
    assert not (ing.db_dir / Layout.processed_dir.path).exists()
    assert not (ing.db_dir / Layout.operations_dir.path).exists()


def test_delete_existing_data_ok_without_operations_dir():
    # Older datasets may have no operations/ yet — clearing must not raise.
    root = _seed_db()
    import shutil
    shutil.rmtree(root / "ds" / Layout.operations_dir.path)
    ing = DatasetIngester(db_dir=root, db_name="ds", repertoire_mapper=None, patient_mapper=None)
    ing._delete_existing_data()          # must not raise
    assert not (ing.db_dir / Layout.processed_dir.path).exists()


def _mixed_locus_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "repertoire_id": ["s1"] * 5,
        "junction": ["tgt"] * 5, "junction_aa": ["CASS"] * 5,
        "v_call": ["TRBV2*01", "TRBV5-1*01", "TRAV1-1*01", "TRAV1-2*01", "XXYV9*01"],
        "j_call": ["TRBJ2-1*01", "TRBJ2-7*01", "TRAJ1*01", "TRAJ2*01", "XXYJ1*01"],
        "duplicate_count": [3, 10, 5, 1, 7],
        "filter_pass": [True] * 5,
    })


def test_partition_by_locus_splits_sorts_and_drops_unknown():
    parts = partition_by_locus(_mixed_locus_frame())
    # TRA + TRB only; the "XX" prefix is outside KNOWN_LOCI and is dropped.
    assert sorted(parts) == ["TRA", "TRB"]
    assert parts["TRB"]["duplicate_count"].to_list() == [10, 3]        # sorted desc
    assert parts["TRA"]["duplicate_count"].to_list() == [5, 1]
    assert "locus" not in parts["TRB"].columns                          # transient col dropped


def test_partition_by_locus_is_lazy_safe():
    parts = partition_by_locus(_mixed_locus_frame().lazy())
    assert all(isinstance(p, pl.LazyFrame) for p in parts.values())     # kind preserved
    assert parts["TRB"].collect()["duplicate_count"].to_list() == [10, 3]


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok:", _name)
