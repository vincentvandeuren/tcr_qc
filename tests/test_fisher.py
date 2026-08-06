"""Fisher / Emerson associated-TCR operation. See docs/fisher_emerson_operation_ideas.md
and docs/superpowers/plans/2026-07-15-fisher-association-operation.md."""
import tempfile
import warnings
from pathlib import Path

import polars as pl
import pytest

from tcr_io.operations.fisher import _fisher, FisherTest, FisherAssociation
from tcr_io import TcrDataset
from tcr_io.structure import (Layout, REQUIRED_DIRS, GENERATED_DIRS, Manifest,
                              repertoire_locus_relpath, repertoire_meta_relpath)
import tcr_io.structure.schema as S
from tcr_io.operations import TabulateByVJ


def test_fisher_matches_scipy_orientation():
    # strong association: clone in all positives, no negatives
    p = _fisher(10, 0, 0, 10, "two-sided")
    assert 0.0 <= p <= 1e-3
    # no association: even split
    assert _fisher(5, 5, 5, 5, "two-sided") == pytest.approx(1.0)


def test_fisher_is_cached_and_handles_bad_table():
    import math
    assert _fisher(1, 2, 3, 4, "two-sided") == _fisher(1, 2, 3, 4, "two-sided")
    assert math.isnan(_fisher(-1, 0, 0, 0, "two-sided"))   # invalid -> nan, no raise


def test_label_expr_boolean_column():
    df = pl.DataFrame({"patient_id": ["a", "b", "c"], "cmv": [True, False, None]})
    out = df.select("patient_id", FisherTest(name="cmv", label_column="cmv", positive=True).label_expr())
    assert out.get_column("label").to_list() == [True, False, None]


def test_label_expr_categorical_positive_negative():
    df = pl.DataFrame({"patient_id": ["a", "b", "c", "d"], "hla": ["B27", "B08", "B27", "other"]})
    # positive = B27, negative unspecified -> every other non-null value is negative
    t = FisherTest(name="b27", label_column="hla", positive="B27")
    assert df.select(t.label_expr()).get_column("label").to_list() == [True, False, True, False]
    # explicit negative subset -> non-listed values drop out (null)
    t2 = FisherTest(name="b27v08", label_column="hla", positive="B27", negative="B08")
    assert df.select(t2.label_expr()).get_column("label").to_list() == [True, False, True, None]


def test_fisher_test_name_must_be_identifier():
    with pytest.raises(ValueError, match="identifier"):
        FisherTest(name="cmv status", label_column="cmv")


def test_associate_builds_standard_table_and_lfc():
    incidence = pl.DataFrame({
        "v_gene": ["TRBV2", "TRBV7-2"],
        "junction_aa": ["CASSPUB", "CASSBG"],
        "j_gene": ["TRBJ2-1", "TRBJ2-1"],
        "n_positive": [2, 2],
        "n_negative": [0, 2],
    })
    op = FisherAssociation(pseudocount=1.0)
    out = op._associate(incidence, total_positive=2, total_negative=2, alternative="two-sided")

    assert out.columns == ["v_gene", "junction_aa", "j_gene", "n_positive", "n_negative",
                           "total_positive", "total_negative", "p_value", "lfc"]
    pub = out.filter(pl.col("junction_aa") == "CASSPUB").to_dicts()[0]
    # standard table for the public clone: a=2,b=0,c=total_pos-2=0,d=total_neg-0=2
    assert pub["p_value"] == pytest.approx(_fisher(2, 0, 0, 2, "two-sided"))
    assert pub["total_positive"] == 2 and pub["total_negative"] == 2
    assert pub["lfc"] > 0                                   # enriched in positives
    bg = out.filter(pl.col("junction_aa") == "CASSBG").to_dicts()[0]
    assert bg["lfc"] == pytest.approx(0.0)                  # even split -> lfc 0


PUBLIC = ("TRBV2*01", "CASSPUBLIC", "TRBJ2-1*01")     # present only in the CMV+ patients
BG = ("TRBV7-2*01", "CASSBG", "TRBJ2-1*01")           # background: present in everyone


def _rep_frame(rid, clones):
    n = len(clones)
    return pl.DataFrame({
        "clonotype_id": list(range(n)),
        "repertoire_id": [rid] * n,
        "junction": ["TGT"] * n,
        "v_call": [c[0] for c in clones],
        "junction_aa": [c[1] for c in clones],
        "j_call": [c[2] for c in clones],
        "duplicate_count": [10] * n,
        "filter_pass": [True] * n,
    }).with_columns(pl.col("clonotype_id").cast(pl.UInt32)).cast(S.REPERTOIRE)


def _make_dataset() -> Path:
    d = Path(tempfile.mkdtemp()) / "ds"
    for sub in REQUIRED_DIRS + GENERATED_DIRS:
        (d / sub).mkdir(parents=True, exist_ok=True)
    layout = {"rep1": ("p1", [PUBLIC, BG]), "rep2": ("p2", [PUBLIC, BG]),
              "rep3": ("p3", [BG]),          "rep4": ("p4", [BG])}
    for rid, (_pid, clones) in layout.items():
        f = d / repertoire_locus_relpath(rid, "TRB")
        f.parent.mkdir(parents=True, exist_ok=True)
        _rep_frame(rid, clones).write_parquet(f)
    reps = list(layout); pats = [layout[r][0] for r in reps]
    pl.DataFrame({"repertoire_id": reps, "source_files": [["x"]] * 4, "patient_id": pats,
                  "n_clonotypes": [2, 2, 1, 1], "n_filtered_clonotypes": [0] * 4,
                  "total_duplicates": [20, 20, 10, 10]}).cast(S.REPERTOIRE_META).write_parquet(
        d / repertoire_meta_relpath("TRB"))
    pl.DataFrame({"patient_id": pats,
                  "patient_repertoires": [[r] for r in reps], "n_repertoires": [1] * 4}
                 ).cast(S.PATIENT_META).write_parquet(d / Layout.patient_meta.path)
    pl.DataFrame({"publication_id": []}, schema=S.PUBLICATION_META).write_ndjson(
        d / Layout.publication_ids.path)
    pl.DataFrame({"dataset_name": ["t"], "created_on": [None], "source": ["x"], "reader": ["r"],
                  "repertoire_mapper": ["m"], "patient_mapper": ["m"]}).cast(S.GENERATION_META
                 ).write_ndjson(d / Layout.generation_meta.path)
    Manifest.current(present_loci=["TRB"]).write(d / Layout.manifest.path)
    return d


def _ds() -> TcrDataset:
    warnings.simplefilter("ignore")
    ds = TcrDataset(_make_dataset())
    ds.set_patient_meta(pl.DataFrame({"patient_id": ["p1", "p2", "p3", "p4"],
                                      "cmv_status": [True, True, False, False]}), mode="replace")
    return ds


def test_fisher_association_public_clone_end_to_end():
    ds = _ds()
    ds.run_operation(TabulateByVJ())
    ds.run_operation(FisherAssociation(
        tests=[FisherTest(name="cmv", label_column="cmv_status", positive=True)]))
    assoc = ds.get_operation_result("fisher_association", "cmv", "TRB")
    assert assoc.columns == ["v_gene", "junction_aa", "j_gene", "n_positive", "n_negative",
                             "total_positive", "total_negative", "p_value", "lfc"]
    pub = assoc.filter(pl.col("junction_aa") == "CASSPUBLIC").to_dicts()[0]
    assert (pub["n_positive"], pub["n_negative"]) == (2, 0)
    assert (pub["total_positive"], pub["total_negative"]) == (2, 2)
    assert pub["lfc"] > 0
    bg = assoc.filter(pl.col("junction_aa") == "CASSBG").to_dicts()[0]
    assert (bg["n_positive"], bg["n_negative"]) == (2, 2)


def test_fisher_autoruns_tabulate_and_keeps_clean_record():
    ds = _ds()                                              # note: TabulateByVJ NOT run first
    ds.run_operation(FisherAssociation(
        tests=[FisherTest(name="cmv", label_column="cmv_status", positive=True)]))
    assert (ds.db_dir / "operations/tabulate_by_vj_gene/operation.json").exists()   # auto-ran
    frec = ds._read_operation_record("fisher_association")
    assert {o.name for o in frec.outputs} == {"cmv"}       # tabulated dir NOT mis-attributed
    assert all(o.kind == "parquet" for o in frec.outputs)


def test_exported_from_operations_package():
    from tcr_io.operations import FisherAssociation as FA, FisherTest as FT
    assert FA.name == "fisher_association"
    assert FT(name="x", label_column="c").name == "x"


def test_multiple_tests_named_outputs_and_namespace():
    ds = _ds()
    ds.run_operation(TabulateByVJ())
    ds.run_operation(FisherAssociation(tests=[
        FisherTest(name="cmv", label_column="cmv_status", positive=True),
        FisherTest(name="cmv_greater", label_column="cmv_status", positive=True, alternative="greater"),
    ]))
    rec = ds._read_operation_record("fisher_association")
    assert {o.name for o in rec.outputs} == {"cmv", "cmv_greater"}
    a = ds.operation_results.fisher_association.TRB.cmv          # tab-completable read-back
    assert "p_value" in a.columns
    # one-sided 'greater' gives a different (<=) p-value for the enriched clone than two-sided
    two = a.filter(pl.col("junction_aa") == "CASSPUBLIC")["p_value"][0]
    one = (ds.operation_results.fisher_association.TRB.cmv_greater
           .filter(pl.col("junction_aa") == "CASSPUBLIC")["p_value"][0])
    assert one <= two


def test_params_serialize_and_rerun():
    ds = _ds()
    ds.run_operation(TabulateByVJ())
    ds.run_operation(FisherAssociation(
        tests=[FisherTest(name="cmv", label_column="cmv_status", positive=True)]))
    rec = ds._read_operation_record("fisher_association")
    assert rec.params["tests"][0]["name"] == "cmv"              # FisherTest serialized as dict
    ds.rerun(FisherAssociation)                                 # reconstructs op from stored dicts
    assert (ds.db_dir / "operations/fisher_association/TRB/cmv.parquet").exists()
