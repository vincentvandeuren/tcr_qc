"""Operations restructure (phases 0-4): write path, skip, retrieval, rerun, meta isolation."""
import json
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tcr_io import TcrDataset
from tcr_io.structure import (
    Layout, REQUIRED_DIRS, GENERATED_DIRS, Manifest, operation_relpath,
    repertoire_relpath, repertoire_meta_relpath,
)
import tcr_io.structure.schema as S
from tcr_io.operations import TestNullOperation, DiversityReport, GeneCountsSummary, TabulateByVJ, RarefactionReport
from tcr_io.operations.base import BaseOperation, OperationResults


def _make_dataset() -> Path:
    d = Path(tempfile.mkdtemp()) / "ds"
    for sub in REQUIRED_DIRS + GENERATED_DIRS:
        (d / sub).mkdir(parents=True, exist_ok=True)

    reps = ["rep_a", "rep_b"]
    (d / "processed_repertoires/TRB").mkdir(parents=True, exist_ok=True)   # locus subdir
    for rid in reps:
        pl.DataFrame({
            "repertoire_id": [rid, rid, rid],
            "junction": ["TGT", "TGC", "TGA"],
            "v_call": ["TRBV2*01", "TRBV7-2*01", "TRBV2*01"],
            "junction_aa": ["CASSF", "CASSY", "CASSL"],
            "j_call": ["TRBJ2-1*01", "TRBJ2-1*01", "TRBJ1-1*01"],
            "duplicate_count": [10, 5, 1],
            "filter_pass": [True, True, False],
        }).cast(S.REPERTOIRE).write_parquet(d / repertoire_relpath(rid, "TRB"))

    pl.DataFrame({
        "repertoire_id": reps, "source_files": [["a"], ["b"]], "patient_id": ["p1", "p1"],
        "n_clonotypes": [2, 2], "n_filtered_clonotypes": [3, 3], "total_duplicates": [16, 16],
    }).cast(S.REPERTOIRE_META).write_parquet(d / repertoire_meta_relpath("TRB"))
    pl.DataFrame({
        "patient_id": ["p1"], "patient_repertoires": [reps], "n_repertoires": [2],
    }).cast(S.PATIENT_META).write_parquet(d / Layout.patient_meta.path)
    pl.DataFrame({"publication_id": []}, schema=S.PUBLICATION_META).write_ndjson(d / Layout.publication_ids.path)
    pl.DataFrame(
        {"dataset_name": ["t"], "created_on": [None], "source": ["x"],
         "reader": ["r"], "repertoire_mapper": ["m"], "patient_mapper": ["m"]}
    ).cast(S.GENERATION_META).write_ndjson(d / Layout.generation_meta.path)
    Manifest.current().write(d / Layout.manifest.path)
    return d


def _ds() -> TcrDataset:
    warnings.simplefilter("ignore")
    return TcrDataset(_make_dataset())


def test_locus_agnostic_write_and_record():
    ds = _ds()
    ds.run_operation(TestNullOperation())
    rec = json.loads((ds.db_dir / "operations/test_null_operation/operation.json").read_text())
    assert rec["status"] == "success"
    assert rec["loci"] is None                                  # locus-agnostic
    assert rec["outputs"] == [{"name": "filter_pass", "path": "filter_pass.parquet",
                               "kind": "parquet", "locus": None}]
    assert (ds.db_dir / "operations/test_null_operation/filter_pass.parquet").exists()


def test_locus_aware_writes_under_locus_subdir():
    ds = _ds()
    ds.run_operation(DiversityReport())                          # supported_loci = ALL_LOCI
    rec = json.loads((ds.db_dir / "operations/diversity_report/operation.json").read_text())
    assert rec["loci"] == ["TRB"]
    assert rec["outputs"][0]["path"] == "TRB/diversity_summary.parquet"
    assert (ds.db_dir / "operations/diversity_report/TRB/diversity_summary.parquet").exists()


def test_retrieval_matches_across_apis():
    ds = _ds()
    ds.run_operation(DiversityReport())
    a = ds.get_operation_result("diversity_report", "diversity_summary", "TRB")
    b = ds.operation_results.diversity_report.TRB.diversity_summary   # locus-explicit
    c = ds.operation_results.diversity_report.diversity_summary       # locus omitted
    assert a.equals(b) and a.equals(c)


def test_retrieval_miss_lists_available():
    ds = _ds()
    ds.run_operation(DiversityReport())
    try:
        ds.get_operation_result("diversity_report", "does_not_exist")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Available" in str(e)


def test_skip_and_force_rerun():
    ds = _ds()
    ds.run_operation(DiversityReport())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds.run_operation(DiversityReport())                     # already done -> skip
    assert any("already complete" in str(x.message) for x in w)
    ds.rerun(DiversityReport)                                   # forced reconstruct + run


def test_gene_counts_names_and_params():
    ds = _ds()
    ds.run_operation(GeneCountsSummary())
    rec = ds._read_operation_record("gene_counts_summary")
    assert rec.params == {}
    assert {o.name for o in rec.outputs} == {"gene_counts", "v_counts", "j_counts", "vj_bias", "vj_long"}


def test_tabulate_unstructured_output():
    ds = _ds()
    ds.run_operation(TabulateByVJ())
    rec = ds._read_operation_record("tabulate_by_vj_gene")
    o = rec.outputs[0]
    assert o.kind == "unstructured" and o.path == "TRB/tabulated"
    tab_dir = ds.get_operation_result("tabulate_by_vj_gene", "tabulated", "TRB")
    assert isinstance(tab_dir, Path) and tab_dir.is_dir()        # a Path, not a frame
    lf = pl.scan_parquet(tab_dir, hive_partitioning=True)        # caller reads it itself
    assert {"v_gene", "j_gene"} <= set(lf.collect_schema().names())


@dataclass
class _UnstructuredProbe(BaseOperation):
    name = "unstructured_probe"
    version = "0.1"
    description = "writes an unstructured dir (with a subdir) plus a serialised frame"
    supported_loci = None                                        # locus-agnostic -> locus=None

    def _run(self, ds, locus=None) -> OperationResults:
        d = ds._operation_output_dir(self, "bundle", locus)
        (d / "a.txt").write_text("hello")
        (d / "sub").mkdir()
        (d / "sub" / "b.txt").write_text("world")
        return OperationResults(outputs={"summary": ds.repertoire_meta(locus).select("repertoire_id")})


def test_unstructured_write_record_and_mixed_outputs():
    ds = _ds()
    ds.run_operation(_UnstructuredProbe())
    rec = ds._read_operation_record("unstructured_probe")
    assert {o.name: o.kind for o in rec.outputs} == {"bundle": "unstructured", "summary": "parquet"}
    assert next(o.path for o in rec.outputs if o.name == "bundle") == "bundle"
    bundle = ds.get_operation_result("unstructured_probe", "bundle")
    assert isinstance(bundle, Path) and (bundle / "a.txt").read_text() == "hello"
    assert (bundle / "sub" / "b.txt").read_text() == "world"
    # the frame declared alongside still reads back as a frame
    assert "repertoire_id" in ds.get_operation_result("unstructured_probe", "summary").columns


def test_unstructured_namespace_returns_path():
    ds = _ds()
    ds.run_operation(_UnstructuredProbe())
    p = ds.operation_results.unstructured_probe.bundle
    assert isinstance(p, Path) and p.is_dir()


def test_unstructured_rerun_empties_dir():
    ds = _ds()
    ds.run_operation(_UnstructuredProbe())
    bundle = ds.get_operation_result("unstructured_probe", "bundle")
    (bundle / "stale.txt").write_text("STALE")
    ds.rerun(_UnstructuredProbe)                                 # forced re-run
    assert not (bundle / "stale.txt").exists()                  # empty-on-allocate wiped it
    assert (bundle / "a.txt").exists()


def test_unstructured_one_dir_per_op_raises():
    @dataclass
    class _TwoDirs(BaseOperation):
        name = "two_dirs"
        version = "0.1"
        description = "requests two unstructured dirs -> must fail"
        supported_loci = None

        def _run(self, ds, locus=None) -> OperationResults:
            ds._operation_output_dir(self, "one", locus)
            ds._operation_output_dir(self, "two", locus)        # second call for same locus
            return OperationResults(outputs={})

    ds = _ds()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds.run_operation(_TwoDirs())                            # run() catches -> failure record
    assert any("one per locus" in str(x.message) for x in w)
    assert ds._read_operation_record("two_dirs").status == "failure"


def test_operation_results_are_real_attributes():
    # Autocomplete depends on the whole `ds.operation_results.<op>[.<locus>]` chain being REAL
    # attributes (no @property / __getattr__ intermediates), else IPython's guarded_eval refuses
    # to traverse it during completion. Guard those structural invariants here.
    from tcr_io.dataset import OperationResultsNamespace
    assert "__getattr__" not in vars(OperationResultsNamespace)   # presence blocks guarded_eval
    ds = _ds()
    ds.run_operation(DiversityReport())
    assert "operation_results" in vars(ds)                        # plain attr, not a @property
    ns = ds.operation_results
    assert "diversity_report" in vars(ns)                         # op handle is a real attribute
    assert "TRB" in vars(ns.diversity_report)                     # locus sub-handle is real too
    # refreshed after a new run so it becomes completable immediately
    ds.run_operation(GeneCountsSummary())
    assert "gene_counts_summary" in vars(ds.operation_results)


def test_operation_results_ipython_completion():
    try:
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        from IPython.core.completer import provisionalcompleter
    except ImportError:
        return                                                    # IPython not installed -> skip
    ds = _ds()
    ds.run_operation(DiversityReport())
    ip = TerminalInteractiveShell.instance()
    ip.user_ns["ds"] = ds
    ip.Completer.evaluation = "limited"                           # the notebook default policy

    def names(text):
        with provisionalcompleter():
            return {c.text.split(".")[-1].lstrip(".") for c in ip.Completer.completions(text, len(text))}

    assert "diversity_report" in names("ds.operation_results.")                     # L1: ops
    assert {"TRB", "diversity_summary"} <= names("ds.operation_results.diversity_report.")  # L2


def test_rarefaction_report():
    ds = _ds()
    ds.run_operation(RarefactionReport(num_points=30, max_depth=1000))
    rec = ds._read_operation_record("rarefaction_report")
    assert rec.loci == ["TRB"]
    assert rec.params == {"num_points": 30, "max_depth": 1000, "extrapolation": True}   # dataclass config
    o = rec.outputs[0]
    assert o.name == "rarefaction_curves" and o.path == "TRB/rarefaction_curves.parquet"
    df = ds.get_operation_result("rarefaction_report", "rarefaction_curves", "TRB")
    assert {"subsampling_depth", "expected_richness", "type", "repertoire_id"} <= set(df.columns)
    assert set(df["repertoire_id"].unique().to_list()) == {"rep_a", "rep_b"}


def test_operations_scan():
    ds = _ds()
    ds.run_operation(TestNullOperation())
    ds.run_operation(DiversityReport())
    assert {r.operation_name for r in ds.operations} == {"test_null_operation", "diversity_report"}


def test_operation_relpath_locus_segment():
    assert operation_relpath("op", "out") == "operations/op/out"
    assert operation_relpath("op", "out", "TRB") == "operations/op/TRB/out"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok:", _name)
