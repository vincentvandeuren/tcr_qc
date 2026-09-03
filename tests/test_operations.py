"""Operations: declaration, fan-out, write path, staleness, retrieval, records."""
import json
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from tcr_io import Dataset
from tcr_io.structure import (
    ARTIFACTS, Artifact, Format, Store, Manifest,
    REPERTOIRE_FILE, REPERTOIRE_META, REPERTOIRE_COUNTS, PATIENT_META, PUBLICATION_IDS,
    GENERATION_META,
)
import tcr_io.structure.schema as S
from tcr_io.operations import (
    NullOperation, DiversityReport, GeneCountsSummary, TabulateByVJ, RarefactionReport,
)
from tcr_io.operations.base import BaseOperation


def _make_dataset() -> Path:
    d = Path(tempfile.mkdtemp()) / "ds"
    Store(d).create_tree(ARTIFACTS)

    reps = ["rep_a", "rep_b"]
    for rid in reps:
        f = d / REPERTOIRE_FILE.template.format(repertoire_id=rid, locus="TRB")
        f.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "repertoire_id": [rid, rid, rid],
            "junction": ["TGT", "TGC", "TGA"],
            "v_call": ["TRBV2*01", "TRBV7-2*01", "TRBV2*01"],
            "junction_aa": ["CASSF", "CASSY", "CASSL"],
            "j_call": ["TRBJ2-1*01", "TRBJ2-1*01", "TRBJ1-1*01"],
            "duplicate_count": [10, 5, 1],
            # v6: the reason the row was dropped, null when it passed. `passed <=>
            # filter_reason.is_null()`, so this is the old `filter_pass` boolean's replacement.
            "filter_reason": [None, None, "min_duplicate_count"],
        }).with_columns(
            clonotype_id=pl.int_range(pl.len(), dtype=pl.UInt32)
        ).cast(S.REPERTOIRE).write_parquet(f)   # locus is the path, not a column

    # v6 splits the old single table: what is true of a repertoire regardless of locus, and
    # what is only true within one. Two writes, two grains.
    pl.DataFrame({
        "repertoire_id": reps, "source_files": [["a"], ["b"]], "patient_id": ["p1", "p1"],
    }).cast(S.REPERTOIRE_META).write_parquet(d / REPERTOIRE_META.template)
    counts = d / REPERTOIRE_COUNTS.template.format(locus="TRB")
    counts.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "repertoire_id": reps, "n_clonotypes": [2, 2], "n_filtered_clonotypes": [1, 1],
        "total_duplicates": [16, 16],
    }).cast(S.REPERTOIRE_COUNTS).write_parquet(counts)
    pl.DataFrame({
        "patient_id": ["p1"], "patient_repertoires": [reps], "n_repertoires": [2],
    }).cast(S.PATIENT_META).write_parquet(d / PATIENT_META.template)
    pl.DataFrame({"publication_id": []}, schema=S.PUBLICATION_META).write_ndjson(
        d / PUBLICATION_IDS.template)
    (d / GENERATION_META.template).write_text(json.dumps(
        {"dataset_name": "t", "created_on": "2026-01-01", "source": "x", "tcrio_version": "t",
         "reader": "r", "repertoire_mapper": "m", "patient_mapper": "m", "filters": {}}))
    Manifest.current().write(Store(d))
    return d


@pytest.fixture
def ds() -> Dataset:
    warnings.simplefilter("ignore")
    return Dataset(_make_dataset())


# --- where a pass writes, and what it records -------------------------------------------

def test_locus_aware_writes_under_a_locus_subdir(ds):
    ds.run_operation(DiversityReport())
    rec = json.loads(
        (ds.db_dir / "operations/diversity_report/TRB/operation.json").read_text())
    assert rec["locus"] == "TRB"
    # the record sits INSIDE the directory it describes, so output paths are relative to it
    assert rec["outputs"][0]["template"] == "diversity_summary.parquet"
    assert (ds.db_dir / "operations/diversity_report/TRB/diversity_summary.parquet").exists()


def test_one_record_per_locus(ds):
    ds.run_operation(NullOperation())
    ds.run_operation(DiversityReport())
    assert {(r.operation_name, r.locus) for r in ds.operations} == {
        ("null_operation", "TRB"), ("diversity_report", "TRB")}


# --- retrieval --------------------------------------------------------------------------

def test_result_reads_back_by_artifact(ds):
    ds.run_operation(DiversityReport())
    a = ds.result(DiversityReport.diversity_summary, locus="TRB").read()
    b = ds.select_locus("TRB").result(DiversityReport.diversity_summary).read()   # bound locus
    assert a.equals(b) and "repertoire_id" in a.columns


def test_raw_result_reads_back_by_name(ds):
    ds.run_operation(DiversityReport())
    by_name = ds.raw_result("diversity_report", "diversity_summary", locus="TRB").read()
    assert by_name.equals(ds.result(DiversityReport.diversity_summary, locus="TRB").read())


def test_raw_result_miss_lists_available(ds):
    ds.run_operation(DiversityReport())
    with pytest.raises(ValueError, match="available"):
        ds.raw_result("diversity_report", "does_not_exist", locus="TRB")


# --- staleness --------------------------------------------------------------------------

def test_second_run_is_reused_and_warns(ds):
    ds.run_operation(DiversityReport())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds.run_operation(DiversityReport())
    assert any("already complete" in str(x.message) for x in w)


def test_rerun_forces_a_rebuild(ds):
    ds.run_operation(DiversityReport())
    first = ds._record("diversity_report", "TRB").ran_at
    recs = ds.rerun(DiversityReport)
    assert [r.status for r in recs] == ["success"]
    assert ds._record("diversity_report", "TRB").ran_at != first


def test_changed_params_are_not_reused(ds):
    ds.run_operation(RarefactionReport(num_points=30, max_depth=1000))
    ds.run_operation(RarefactionReport(num_points=10, max_depth=1000))    # different identity
    assert ds._record("rarefaction_report", "TRB").params["num_points"] == 10


def test_a_recordless_directory_is_cleared_before_it_is_reused(ds):
    """Files with no record are a dead run's leftovers, and they must not be adopted.

    `_run` writes the outputs and the record is written after, so a crash between the two
    leaves outputs behind with nothing describing them. `_observe` GLOBS, so without a clear
    the next pass finds those orphans and records them as its own — which is how a
    parameterised output for a parameter this pass never computed ends up in its record.

    This is why the staleness verdict is a boolean and not a three-way: the old `RUN` state
    meant "no record, so compute without clearing", and that state was this bug.
    """
    @dataclass
    class _Keyed(BaseOperation):
        name, version, description = "keyed", "0.1", "one table per key"
        supported_loci = frozenset({"TRB"})
        table = Artifact("{k}.parquet", Format.PARQUET)
        keys: tuple = ("a",)

        def _run(self, ds, out):
            for k in self.keys:
                out(self.table, k=k).write(pl.DataFrame({"x": [1]}))

    ds.run_operation(_Keyed(keys=("a",)))
    outdir = ds.db_dir / "operations/keyed/TRB"
    (outdir / "operation.json").unlink()          # the crash: outputs kept, record lost

    recs = ds.run_operation(_Keyed(keys=("b",)))

    assert {o.template for o in recs[0].outputs} == {"b.parquet"}   # not a.parquet
    assert not (outdir / "a.parquet").exists()


def test_a_failing_locus_records_the_failure(ds):
    @dataclass
    class _Boom(BaseOperation):
        name, version, description = "boom", "0.1", "always raises"
        supported_loci = frozenset({"TRB"})
        table = Artifact("table.parquet", Format.PARQUET)

        def _run(self, ds, out):
            raise RuntimeError("nope")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds.run_operation(_Boom())
    assert any("failed on locus=TRB" in str(x.message) for x in w)
    rec = ds._record("boom", "TRB")
    assert rec.status == "failure" and "nope" in rec.error


# --- output declaration -----------------------------------------------------------------

def test_gene_counts_records_every_declared_output(ds):
    ds.run_operation(GeneCountsSummary())
    rec = ds._record("gene_counts_summary", "TRB")
    assert rec.params == {}
    assert {o.name for o in rec.outputs} == {
        "gene_counts", "v_counts", "j_counts", "vj_bias", "vj_long"}


def test_a_declared_output_that_is_never_written_is_an_error(ds):
    @dataclass
    class _Forgetful(BaseOperation):
        name, version, description = "forgetful", "0.1", "declares two, writes one"
        written = Artifact("written.parquet", Format.PARQUET)
        forgotten = Artifact("forgotten.parquet", Format.PARQUET)

        def _run(self, ds, out):
            out(self.written).write(ds.repertoire_meta.select("repertoire_id"))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ds.run_operation(_Forgetful())
    assert "forgotten" in ds._record("forgetful", "TRB").error


def test_an_inherited_output_is_refused():
    # `ds.result()` resolves an output's directory from `owner.name`, so a shared declaration
    # would file every subclass's results under the base's folder.
    class _Base(BaseOperation):
        name, version, description = "shared_base", "0.1", ""
        shared = Artifact("shared.parquet", Format.PARQUET)

        def _run(self, ds, out):
            pass

    class _Sub(_Base):
        name = "shared_sub"

    with pytest.raises(TypeError, match="inherits output"):
        _Sub.artifacts()
# --- directory outputs ------------------------------------------------------------------

def test_tabulate_writes_a_scannable_directory(ds):
    ds.run_operation(TabulateByVJ())
    rec = ds._record("tabulate_by_vj_gene", "TRB")
    o = rec.outputs[0]
    assert o.name == "tabulated" and o.format == "PARQUET_DIR" and o.template == "tabulated"

    handle = ds.result(TabulateByVJ.tabulated, locus="TRB")
    assert handle.path().is_dir()
    names = handle.scan().collect_schema().names()               # no read(); scan only
    assert {"v_gene", "j_gene"} <= set(names)


def test_rerunning_a_directory_output_clears_stale_partitions(ds):
    ds.run_operation(TabulateByVJ())
    tab = ds.result(TabulateByVJ.tabulated, locus="TRB").path()
    (tab / "stale.txt").write_text("STALE")
    ds.rerun(TabulateByVJ)
    assert not (tab / "stale.txt").exists()
    assert any(tab.iterdir())


# --- fan-out ----------------------------------------------------------------------------

def test_a_bound_dataset_fans_out_over_its_locus_alone(ds):
    seen = []

    @dataclass
    class _Spy(BaseOperation):
        name, version, description = "spy", "0.1", "records the loci it was run on"
        supported_loci = frozenset({"TRA", "TRB"})
        table = Artifact("table.parquet", Format.PARQUET)

        def _run(self, ds, out):
            seen.append(ds.locus)
            out(self.table).write(ds.repertoire_meta().select("repertoire_id"))

    ds.select_locus("TRB").run_operation(_Spy())
    assert seen == ["TRB"]        # TRA is supported but not offered by the binding


def test_unsupported_loci_are_skipped(ds):
    @dataclass
    class _TraOnly(BaseOperation):
        name, version, description = "tra_only", "0.1", "supports a locus this dataset lacks"
        supported_loci = frozenset({"TRA"})
        table = Artifact("table.parquet", Format.PARQUET)

        def _run(self, ds, out):
            raise AssertionError("must not run")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert ds.run_operation(_TraOnly()) == []
    assert any("already complete" in str(x.message) for x in w)


# --- a real op end to end ---------------------------------------------------------------

def test_rarefaction_report(ds):
    ds.run_operation(RarefactionReport(num_points=30, max_depth=1000))
    rec = ds._record("rarefaction_report", "TRB")
    assert rec.params == {"num_points": 30, "max_depth": 1000, "extrapolation": True}
    assert [o.name for o in rec.outputs] == ["rarefaction_curves"]
    df = ds.result(RarefactionReport.rarefaction_curves, locus="TRB").read()
    assert {"subsampling_depth", "expected_richness", "type", "repertoire_id"} <= set(df.columns)
    assert set(df["repertoire_id"].unique().to_list()) == {"rep_a", "rep_b"}


# --- map_repertoires --------------------------------------------------------------------

def test_map_repertoires_tags_and_concats(ds):
    out = ds.select_locus("TRB").map_repertoires(lambda df: df.select(pl.len().alias("n")))
    out = out.collect() if isinstance(out, pl.LazyFrame) else out
    assert set(out["repertoire_id"].to_list()) == {"rep_a", "rep_b"}
    assert out.height == 2 and "n" in out.columns


def test_map_repertoires_concat_false_returns_parts(ds):
    parts = ds.select_locus("TRB").map_repertoires(
        lambda df: df.select(pl.len().alias("n")), concat=False)
    assert isinstance(parts, list) and len(parts) == 2


def test_map_repertoires_empty_needs_schema(ds):
    # Bound straight to a locus with no partitions on disk. `select_locus` would reject "TRA"
    # (it is not a locus of this dataset); the constructor binds whatever it is given, which is
    # what makes the empty-repertoire-set path reachable at all.
    empty_ds = Dataset(ds.db_dir, locus="TRA")
    sch = pl.Schema({"repertoire_id": pl.Utf8, "n": pl.UInt32})
    empty = empty_ds.map_repertoires(lambda df: df.select(pl.len().alias("n")), schema=sch)
    empty = empty.collect() if isinstance(empty, pl.LazyFrame) else empty
    assert empty.height == 0 and empty.columns == ["repertoire_id", "n"]
    with pytest.raises(ValueError):
        empty_ds.map_repertoires(lambda df: df.select(pl.len().alias("n")))


# --- filtering report -------------------------------------------------------------------

def test_filtering_report_unassigned_summary(ds):
    from tcr_io.operations.filtering_report import FilteringReport
    # add an _unassigned partition for rep_a: one null v_call, one null j_call
    f = ds.db_dir / REPERTOIRE_FILE.template.format(repertoire_id="rep_a", locus="_unassigned")
    f.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "repertoire_id": ["rep_a"] * 2, "junction": ["TGA", "TGG"],
        "v_call": [None, "TRBV2*01"], "junction_aa": ["CASSL", "CASSX"],
        "j_call": ["TRBJ2-1*01", None], "duplicate_count": [1, 1],
        "filter_reason": ["null_v", "null_j"],
    }).with_columns(
        clonotype_id=pl.int_range(pl.len(), dtype=pl.UInt32)
    ).cast(S.REPERTOIRE).write_parquet(f)

    ds.run_operation(FilteringReport())
    # Assignment failures live under the `_unassigned` pseudo-locus, same shape as a quality
    # report: a wide first-failure summary + per-reason top tables + totals.
    summ = ds.result(FilteringReport.filter_summary, locus="_unassigned").read()
    row = summ.filter(pl.col("repertoire_id") == "rep_a").to_dicts()[0]
    assert (row["null_v"], row["null_j"]) == (1, 1)
    totals = ds.result(FilteringReport.filter_reason_totals, locus="_unassigned").read()
    t = dict(zip(totals["reason"].to_list(), totals["n_failed_rows"].to_list()))
    assert t["null_v"] == 1 and t["null_j"] == 1


def test_filtering_report_quality_population(ds):
    from tcr_io.operations.filtering_report import FilteringReport
    # Overwrite rep_a's TRB partition with rows whose failure reasons are known. Since v0.6
    # the report READS `filter_reason` rather than rebuilding it, so the fixture states the
    # reason ingestion would have written — including the first-failure attribution below.
    f = ds.db_dir / REPERTOIRE_FILE.template.format(repertoire_id="rep_a", locus="TRB")
    pl.DataFrame({
        "repertoire_id": ["rep_a"] * 4,
        "junction":      ["TGTGCC", "TGTGCC",   "TGTGCC",      None],
        "v_call":        ["TRBV2*01", "TRBV2*01", "TRBV999*01", "TRBV2*01"],
        "junction_aa":   ["CASSLGYEQYF", "XXX",   "CASSLGYEQYF", "XXX"],
        "j_call":        ["TRBJ2-1*01"] * 4,
        "duplicate_count": [10, 1, 1, 1],
        # row0 passes (null reason); rows 1-3 carry the first filter that rejected them. The
        # last row is null junction + "XXX" junction_aa: ingest attributes it to null_junction,
        # which comes first in the default set, NOT to invalid_junction_aa.
        "filter_reason": [None, "invalid_junction_aa", "invalid_v_call", "null_junction"],
    }).with_columns(
        clonotype_id=pl.int_range(pl.len(), dtype=pl.UInt32)
    ).cast(S.REPERTOIRE).write_parquet(f)

    ds.run_operation(FilteringReport())

    summ = ds.result(FilteringReport.filter_summary, locus="TRB").read()
    row = summ.filter(pl.col("repertoire_id") == "rep_a").to_dicts()[0]
    assert row["invalid_junction_aa"] == 1
    assert row["invalid_v_call"] == 1
    assert row["null_junction"] == 1

    top_aa = ds.result(FilteringReport.filter_top, locus="TRB",
                       reason="invalid_junction_aa").read()
    assert top_aa.filter(pl.col("junction_aa") == "XXX")["count"].to_list() == [1]
    top_v = ds.result(FilteringReport.filter_top, locus="TRB", reason="invalid_v_call").read()
    assert top_v.filter(pl.col("v_call") == "TRBV999*01")["count"].to_list() == [1]

    # A reason that never fired gets NO table. v0.5 emitted an empty one, because it rebuilt
    # the report from the filter set and so knew every reason that could have fired; v0.6 reads
    # the stored `filter_reason`, which only names the ones that did. The runner allows it —
    # `_observe` exempts parameterised artifacts, since "no filter reason fired" is a
    # legitimate zero — and the legend below still says which reasons the locus saw.
    assert not ds.result(FilteringReport.filter_top, locus="TRB",
                         reason="invalid_j_call").exists()

    legend = ds.result(FilteringReport.filter_legend, locus="TRB").read()
    assert {"reason", "description", "group_col"} <= set(legend.columns)
    assert "imgt_functional" not in legend["reason"].to_list()   # expanded into sub-reasons
    assert "invalid_v_call" in legend["reason"].to_list()
    assert legend["reason"].n_unique() == legend.height
    # assignment reasons (null_v/null_j) are NOT in a real locus's quality legend — they are
    # reported only on the _unassigned partition (population separation).
    assert "null_v" not in legend["reason"].to_list()
