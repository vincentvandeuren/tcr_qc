"""Single-cell support: cell_id list aggregation, per-(repertoire, locus) clonotype_id, and the
clone_to_cell split (referential integrity + determinism)."""
import tempfile
from pathlib import Path

import polars as pl

from tcr_io.grouper import Grouper
from tcr_io.readers import CellrangerReader, AirrReader
from tcr_io.ingestion import DatasetIngester
from tcr_io.mappers import RegexMapper
from tcr_io.dataset import TcrDataset
from tcr_io.structure import Layout, CLONE_TO_CELL


# --- fixtures ---------------------------------------------------------------

# 3 cells, paired TRB+TRA; b1/b2 share a TRB clonotype; all three share a TRA clonotype.
_CELLRANGER_ROWS = [
    # barcode, is_cell, v_gene, j_gene, cdr3_nt, cdr3, umis
    ("b1", True, "TRBV2", "TRBJ2-1", "TGTGCCAGCAGCCTG", "CASSL", 5),
    ("b2", True, "TRBV2", "TRBJ2-1", "TGTGCCAGCAGCCTG", "CASSL", 3),
    ("b3", True, "TRBV7-2", "TRBJ2-1", "TGTGCCAGCAGCTAC", "CASSY", 2),
    ("b1", True, "TRAV1-1", "TRAJ1", "TGTGCCGTGCGC", "CAVR", 4),
    ("b2", True, "TRAV1-1", "TRAJ1", "TGTGCCGTGCGC", "CAVR", 4),
    ("b3", True, "TRAV1-1", "TRAJ1", "TGTGCCGTGCGC", "CAVR", 1),
]


def _cellranger_src() -> Path:
    src = Path(tempfile.mkdtemp())
    pl.DataFrame(_CELLRANGER_ROWS,
                 schema=["barcode", "is_cell", "v_gene", "j_gene", "cdr3_nt", "cdr3", "umis"],
                 orient="row").write_csv(src / "sample1.csv")
    return src


def _ingest(src: Path, reader) -> TcrDataset:
    db = Path(tempfile.mkdtemp())
    ing = DatasetIngester(
        db_dir=db, db_name="ds", reader=reader,
        repertoire_mapper=RegexMapper(r"(sample\d+)", group=1),
        patient_mapper=RegexMapper(r"(sample\d+)", group=1), allow_overwrite=True,
    )
    return ing.run(src)


# --- grouper unit -----------------------------------------------------------

def test_grouper_list_cols_aggregates_cell_id_and_sums_umis():
    df = pl.DataFrame({
        "v_call": ["TRBV2*01", "TRBV2*01", "TRBV7*01"],
        "junction": ["AAA", "AAA", "CCC"],
        "j_call": ["TRBJ1*01", "TRBJ1*01", "TRBJ1*01"],
        "junction_aa": ["CAA", "CAA", "CCC"],
        "duplicate_count": [2, 3, 1],
        "cell_id": ["b1", "b2", "b3"],
    })
    g = Grouper(by="clonotype_nt").run(df)
    shared = g.filter(pl.col("junction") == "AAA").to_dicts()[0]
    assert sorted(shared["cell_id"]) == ["b1", "b2"]   # cell_id -> List
    assert shared["duplicate_count"] == 5              # UMIs summed (sum_cols)
    # a bulk frame (no cell_id) is unaffected: no list column appears
    assert "cell_id" not in Grouper().run(df.drop("cell_id")).columns


# --- single-cell ingestion --------------------------------------------------

def test_single_cell_clonotype_id_and_referential_integrity():
    ds = _ingest(_cellranger_src(), CellrangerReader())
    assert sorted(ds._present_loci()) == ["TRA", "TRB"]

    # per-(repertoire, locus) clonotype_id: contiguous 0..N-1 within each locus
    assert ds.read_repertoire("sample1", locus="TRA")["clonotype_id"].to_list() == [0]
    assert sorted(ds.read_repertoire("sample1", locus="TRB")["clonotype_id"].to_list()) == [0, 1]
    # b1+b2 collapsed (UMIs 5+3=8); all TRA reads collapsed (4+4+1=9)
    assert set(ds.read_repertoire("sample1", locus="TRB")["duplicate_count"].to_list()) == {8, 2}

    ctc = ds.clone_to_cell("sample1")
    assert ctc.schema == dict(CLONE_TO_CELL)
    assert ctc.height == 6                                   # 2 + 1 (TRB) + 3 (TRA)
    # every (locus, clonotype_id) in the map joins back to exactly one clonotype row
    for locus in ds._present_loci():
        clono = ds.read_repertoire("sample1", locus=locus).select("clonotype_id").unique()
        mapped = ctc.filter(pl.col("locus") == locus).select("clonotype_id").unique()
        assert mapped.join(clono, on="clonotype_id", how="anti").height == 0
    # a paired cell appears once per locus (b1 in both its TRA and TRB clonotype)
    assert set(ctc.filter(pl.col("cell_id") == "b1")["locus"].to_list()) == {"TRA", "TRB"}
    # cell_id never leaks into the clonotype table
    assert "cell_id" not in ds.read_repertoire("sample1", locus="TRB").columns


def test_airr_cell_id_drives_single_cell_path():
    # An AIRR file that happens to carry `cell_id` is picked up via optional_col_map -> single-cell.
    src = Path(tempfile.mkdtemp())
    pl.DataFrame({
        "cell_id": ["c1", "c2", "c3"],
        "v_call": ["TRBV2*01", "TRBV2*01", "TRAV1-1*01"],
        "j_call": ["TRBJ2-1*01", "TRBJ2-1*01", "TRAJ1*01"],
        "junction": ["TGTGCCAGCAGCCTG", "TGTGCCAGCAGCCTG", "TGTGCCGTGCGC"],
        "junction_aa": ["CASSL", "CASSL", "CAVR"],
        "duplicate_count": [5, 3, 2],
    }).write_csv(src / "sample1.tsv", separator="\t")

    ds = _ingest(src, AirrReader())
    ctc = ds.clone_to_cell("sample1")
    assert ctc.height == 3 and set(ctc["locus"].to_list()) == {"TRA", "TRB"}
    # c1+c2 collapse into one TRB clonotype carrying both cells
    assert sorted(ctc.filter(pl.col("locus") == "TRB")["cell_id"].to_list()) == ["c1", "c2"]


def test_single_cell_reingest_is_deterministic():
    src = _cellranger_src()
    a = _ingest(src, CellrangerReader()).clone_to_cell("sample1").sort("locus", "clonotype_id", "cell_id")
    b = _ingest(src, CellrangerReader()).clone_to_cell("sample1").sort("locus", "clonotype_id", "cell_id")
    assert a.equals(b)


# --- bulk path is unaffected apart from the new column -----------------------

def test_bulk_gets_clonotype_id_and_no_clone_to_cell():
    src = Path(tempfile.mkdtemp())
    pl.DataFrame({
        "v_call": ["TRBV2*01", "TRBV7-2*01", "TRBV2*01"],
        "j_call": ["TRBJ2-1*01", "TRBJ2-1*01", "TRBJ2-1*01"],
        "junction": ["TGTGCCAGCAGCCTG", "TGTGCCAGCAGCTAC", "TGTGCCAGCAGCGGG"],
        "junction_aa": ["CASSL", "CASSY", "CASSG"],
        "duplicate_count": [10, 5, 1],
    }).write_csv(src / "sample1.tsv", separator="\t")

    ds = _ingest(src, AirrReader())
    clono = ds.read_repertoire("sample1", locus="TRB")
    assert clono["clonotype_id"].dtype == pl.UInt32
    assert sorted(clono["clonotype_id"].to_list()) == [0, 1, 2]     # contiguous within the locus
    assert not (ds.db_dir / Layout.clone_to_cell_dir.path).exists()  # lazy: bulk never creates it
    assert ds.clone_to_cell().height == 0                            # empty accessor for bulk
