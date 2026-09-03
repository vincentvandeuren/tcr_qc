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
    Filterer (byte-identical, verified during the refactor). Guards against silent drift.

    `pass_expr()` is unaliased — it is a bare boolean, and the caller names it — so the alias
    is the test's job. The second assertion pins the invariant the v6 schema rests on:
    `passed <=> filter_reason.is_null()`. They are two expressions and could drift apart."""
    expected = [True, False, False, False, False, False]
    fs = FilterSet.named("default_trb")
    assert _FIXTURE.select(fs.pass_expr().alias("filter_pass"))["filter_pass"].to_list() == expected
    assert _FIXTURE.select(fs.reason_expr())["filter_reason"].is_null().to_list() == expected


def test_named_and_available():
    """`named()` is the whole reconstruction path — a preset name is what re-runs an ingest,
    and `available()` is how you find one. There is no rebuild-from-filter-names any more."""
    assert "default_trb" in FilterSet.available()
    assert FilterSet.named("default_trb")._preset_name == "default_trb"
    with pytest.raises(KeyError, match="available"):
        FilterSet.named("does_not_exist")
    assert not hasattr(FilterSet, "from_names")


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
    assert "filter_reason" in out.columns          # v6: the reason, not a boolean
    # Rows that could not be placed land in `_unassigned` and carry the ASSIGNMENT reason they
    # failed on — not a quality reason they were never tested against. That distinction is the
    # reason those rows are kept at all.
    row = out.filter(pl.col("locus") == UNASSIGNED)
    assert row["filter_reason"].to_list() == ["null_v", "null_j"]
    assert row["filter_reason"].is_null().sum() == 0        # every one is excluded


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
    # A locus nobody supplied a filter set for is excluded wholesale, and says so rather than
    # borrowing a quality reason. TRB passes, so its reason is null.
    assert dict(zip(out["locus"], out["filter_reason"])) == {"TRA": "locus_not_accepted",
                                                            "TRB": None}


def test_per_locus_duplicate_locus_raises():
    with pytest.raises(ValueError):
        PerLocusFilterSet({("TRA", "TRB"): FilterSet.named("default_trb"),
                           ("TRB",): FilterSet.named("default_trb")})
