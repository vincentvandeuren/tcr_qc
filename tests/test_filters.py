"""Phase 0/1 of the filter revamp: filter object model, FilterSet, presets, PerLocusFilterSet."""
import polars as pl
import pytest

from tcr_io.filters import (
    FILTER_REGISTRY,
    BaseFilter,
    FilterSet,
    ImgtFunctionalFilter,
    NullJunctionFilter,
    NullVFilter,
    PerLocusFilterSet,
    ValidJunctionAaFilter,
)
from tcr_io.expressions import UNASSIGNED


# A fixture with variety: valid rows, nulls, an invalid gene, malformed junction_aa.
_FIXTURE = pl.DataFrame({
    "v_call":      ["TRBV2*01", None,        "TRBV7-2*01", "TRBV999*01", "TRBV2*01",   "TRBV2*01"],
    "j_call":      ["TRBJ2-1*01", "TRBJ2-1*01", None,      "TRBJ2-1*01", "TRBJ1-1*01", "TRBJ2-1*01"],
    "junction":    ["TGTGCC",   "TGTGCC",    "TGTGCC",     "TGTGCC",     None,         "TGTGCC"],
    "junction_aa": ["CASSLGYEQYF", "CASSLGYEQYF", "CASSLGYEQYF", "CASSLGYEQYF", "CASSLGYEQYF", "XX"],
})


def test_filter_registry_populated():
    for name in ["null_v", "null_j", "null_junction", "null_junction_aa",
                 "valid_junction_aa", "imgt_functional"]:
        assert name in FILTER_REGISTRY
    # StructFilter is abstract-ish (no name) and must not register.
    assert "" not in FILTER_REGISTRY


def test_every_registered_filter_has_description():
    for name, cls in FILTER_REGISTRY.items():
        assert cls.description.strip(), f"{name} missing description"


def test_pass_expr_semantics():
    out = _FIXTURE.select(
        NullVFilter().pass_expr().alias("v"),
        ValidJunctionAaFilter().pass_expr().alias("aa"),
    )
    assert out["v"].to_list() == [True, False, True, True, True, True]
    # row 5 has junction_aa "XX" -> invalid
    assert out["aa"].to_list()[-1] is False


def test_imgt_sub_reports():
    reports = ImgtFunctionalFilter().reports()
    assert [r.name for r in reports] == [
        "invalid_v_call", "non_functional_v_call", "invalid_j_call", "non_functional_j_call",
    ]
    assert all(r.description for r in reports)
    assert [r.group_col for r in reports] == ["v_call", "v_call", "j_call", "j_call"]


def test_filterset_golden_snapshot():
    """FilterSet.pass_expr on the fixture must match the snapshot captured from the legacy
    Filterer (byte-identical, verified during the refactor). Guards against silent drift."""
    expected = [True, False, False, False, False, False]
    new = _FIXTURE.select(FilterSet.named("default_trb").pass_expr())["filter_pass"]
    assert new.to_list() == expected


def test_from_names_roundtrip():
    names = FilterSet.named("default_trb").names
    assert FilterSet.from_names(names).names == names


def test_named_and_available():
    assert "default_trb" in FilterSet.available()
    assert FilterSet.named("default_trb")._preset_name == "default_trb"
    with pytest.raises(KeyError):
        FilterSet.named("does_not_exist")


def test_reasons_flattened_and_ordered():
    reasons = FilterSet.named("default_trb").reasons()
    names = [r.name for r in reasons]
    # scalar filters keep their own name; imgt expands into 4 sub-reports, in order
    assert names == [
        "null_v", "null_j", "null_junction", "null_junction_aa", "invalid_junction_aa",
        "invalid_v_call", "non_functional_v_call", "invalid_j_call", "non_functional_j_call",
    ]


def test_per_locus_filter_set_assigns_locus_and_forces_unassigned():
    pl_fs = PerLocusFilterSet.uniform(FilterSet.named("default_trb"))
    out = pl_fs.run(_FIXTURE.lazy()).collect()
    assert "locus" in out.columns
    assert "filter_pass" in out.columns
    # null v_call (row 1) -> _unassigned partition -> forced False
    row = out.filter(pl.col("locus") == UNASSIGNED)
    assert row["filter_pass"].to_list() == [False] * row.height


def test_per_locus_explicit_mapping():
    tcr = FilterSet.named("default_trb")
    override = FilterSet([NullVFilter(), NullJunctionFilter()])
    pl_fs = PerLocusFilterSet({("TRA", "TRB"): tcr, ("IGH",): override})
    assert pl_fs.for_locus("TRB") is tcr
    assert pl_fs.for_locus("TRA") is tcr
    assert pl_fs.for_locus("IGH") is override
    assert pl_fs.for_locus("TRG") is None          # not listed -> not accepted
    assert pl_fs.accepted_loci == ["IGH", "TRA", "TRB"]


def test_per_locus_unlisted_chain_fails():
    # only TRB accepted; a TRA row (valid) must be forced to fail, TRB row passes
    pl_fs = PerLocusFilterSet({("TRB",): FilterSet.named("default_trb")})
    df = pl.DataFrame({
        "v_call":      ["TRBV2*01",  "TRAV1-1*01"],
        "j_call":      ["TRBJ2-1*01", "TRAJ2*01"],
        "junction":    ["TGTGCCAGCAGCTTAGGGTATGAACAGTACTTC"] * 2,
        "junction_aa": ["CASSLGYEQYF", "CASSLGYEQYF"],
    })
    out = pl_fs.run(df.lazy()).collect().sort("locus")
    assert dict(zip(out["locus"], out["filter_pass"])) == {"TRA": False, "TRB": True}


def test_per_locus_duplicate_locus_raises():
    with pytest.raises(ValueError):
        PerLocusFilterSet({("TRA", "TRB"): FilterSet.named("default_trb"),
                           ("TRB",): FilterSet.named("default_trb")})
