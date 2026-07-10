"""Ingestion behaviours that don't need the full reader/mapper pipeline."""
import json
import tempfile
from pathlib import Path

import polars as pl

from tcr_io.ingestion import DatasetIngester
from tcr_io.expressions import assign_locus, UNASSIGNED
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
        "v_call": ["TRBV2*01", "TRAV1-1*01", "XXYV9*01", "TRBV5*01", None],
        "j_call": ["TRBJ2-1*01", "TRAJ1*01", "XXYJ1*01", "TRAJ2*01", "TRBJ1-1*01"],
    })


def test_assign_locus_maps_known_and_unassigned():
    out = _mixed_locus_frame().with_columns(assign_locus())["locus"].to_list()
    # TRB, TRA are known; "XX" prefix is unknown; TRB/TRA prefix mismatch; null v_call -> all _unassigned
    assert out == ["TRB", "TRA", UNASSIGNED, UNASSIGNED, UNASSIGNED]


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok:", _name)
