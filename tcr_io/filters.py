from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

import polars as pl

from .expressions import (
    KNOWN_LOCI,
    assign_locus,
    is_functional_tcr,
    is_valid_junction_aa,
)
from .structure.version import IMGT_VERSION


# name -> class, so a FilterSet can be rebuilt from serialised names (provenance).
FILTER_REGISTRY: dict[str, type["BaseFilter"]] = {}


@dataclass(eq=False)
class FilterReport:
    """One reporting unit: how to surface the top offending values among FAILED rows.

    `mask` is a boolean expression that is True on rows this reason explains (evaluated on the
    already-failed subset); `group_col` is the column whose most frequent values are tabulated.
    """
    name: str
    mask: pl.Expr
    group_col: str
    description: str = ""


class BaseFilter(ABC):
    """A single named filter. `pass_expr()` is boolean: True/null => pass, False => fail.

    Concrete subclasses set a class-level `name` (and usually `description`/`group_col`) and are
    auto-registered in `FILTER_REGISTRY`. Abstract intermediates (e.g. `StructFilter`) leave
    `name` unset and are skipped.
    """
    name: str = ""               # filter identity (registry / provenance key; a pass-condition name)
    reason_name: str = ""        # FAILURE-reason label for the report; defaults to `name`
    description: str = ""        # human-readable, surfaced by the report's filter_legend
    group_col: str = ""          # column whose top values the default report tabulates

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if getattr(cls, "name", ""):
            FILTER_REGISTRY[cls.name] = cls

    @abstractmethod
    def pass_expr(self) -> pl.Expr:
        """Boolean; True/null => pass, False => fail."""

    def filter_expr(self) -> pl.Expr:
        return self.pass_expr().alias(f"filter_{self.name}")

    def reports(self) -> List[FilterReport]:
        """Default: one report keyed by the failure-reason label, grouping by `group_col`.

        Most filter names already describe the failure (null_v, null_junction, ...) so
        `reason_name` defaults to `name`; filters whose name is a *pass* condition
        (e.g. valid_junction_aa) set `reason_name` to the negative (invalid_junction_aa).
        """
        return [FilterReport(
            name=self.reason_name or self.name,
            mask=~self.pass_expr().fill_null(True),
            group_col=self.group_col,
            description=self.description,
        )]


class StructFilter(BaseFilter):
    """One pass/fail column from a struct-returning expr with several boolean fields.

    `pass_expr` ANDs `filter_fields`; subclasses override `reports()` to expose finer sub-reports
    (built in the method because they reference `self.struct_expr`).
    """
    struct_expr: pl.Expr = pl.lit(None)
    filter_fields: tuple = ()

    def pass_expr(self) -> pl.Expr:
        expr = pl.lit(True)
        for f in self.filter_fields:
            expr = expr & self.struct_expr.struct.field(f)
        return expr


# --- concrete filters (names match the legacy struct-field aliases) --------

class NullVFilter(BaseFilter):
    name = "null_v"
    group_col = "v_call"
    description = "V gene call is missing (null)."
    def pass_expr(self) -> pl.Expr:
        return pl.col("v_call").is_not_null()


class NullJFilter(BaseFilter):
    name = "null_j"
    group_col = "j_call"
    description = "J gene call is missing (null)."
    def pass_expr(self) -> pl.Expr:
        return pl.col("j_call").is_not_null()


class NullJunctionFilter(BaseFilter):
    name = "null_junction"
    group_col = "junction"
    description = "Nucleotide junction sequence is missing (null)."
    def pass_expr(self) -> pl.Expr:
        return pl.col("junction").is_not_null()


class NullJunctionAAFilter(BaseFilter):
    name = "null_junction_aa"
    group_col = "junction_aa"
    description = "Amino-acid junction sequence is missing (null)."
    def pass_expr(self) -> pl.Expr:
        return pl.col("junction_aa").is_not_null()


class ValidJunctionAaFilter(BaseFilter):
    name = "valid_junction_aa"          # registry/provenance key (pass-condition, matches legacy)
    reason_name = "invalid_junction_aa"  # failure label for the report
    group_col = "junction_aa"
    description = ("Amino-acid junction is malformed: wrong length, a non-amino-acid character, "
                  "or a non-canonical start/end residue.") # @claude can you make this less concise and more explicit / exact?
    def pass_expr(self) -> pl.Expr:
        return is_valid_junction_aa()


class ImgtFunctionalFilter(StructFilter):
    name = "imgt_functional"
    group_col = "v_call"
    description = (f"Looks each V and J gene up in the IMGT reference database (version:{IMGT_VERSION}). Fails if a gene is "
                  "not found (invalid notation / not a real gene) or is found but flagged "
                  "non-functional (pseudogene / ORF).")
    struct_expr = is_functional_tcr(pl.col("v_call"), pl.col("j_call"), organism="human")
    filter_fields = ("v_valid_imgt", "v_func_imgt", "j_valid_imgt", "j_func_imgt")

    def reports(self) -> List[FilterReport]:
        F = lambda x: self.struct_expr.struct.field(x)  # noqa: E731 (local, readability)
        return [
            FilterReport("invalid_v_call", ~F("v_valid_imgt"), "v_call",
                         "V gene is not in the IMGT reference database."),
            FilterReport("non_functional_v_call", F("v_valid_imgt") & ~F("v_func_imgt"), "v_call",
                         "V gene is a real IMGT gene but flagged non-functional."),
            FilterReport("invalid_j_call", ~F("j_valid_imgt"), "j_call",
                         "J gene is not in the IMGT reference database."),
            FilterReport("non_functional_j_call", F("j_valid_imgt") & ~F("j_func_imgt"), "j_call",
                         "J gene is a real IMGT gene but flagged non-functional."),
        ]


# ---------------------------------------------------------------------------
# FilterSet + named-preset registry (Phase 1)
# ---------------------------------------------------------------------------

# name -> builder, for named presets (e.g. "default_trb", "single_cell").
FILTER_SET_REGISTRY: dict[str, Callable[[], "FilterSet"]] = {}


def register_filter_set(name: str):
    """Register a named `FilterSet` builder. Usage::

        @register_filter_set("default_trb")
        def _default_trb() -> FilterSet:
            return FilterSet([...])
    """
    def deco(builder: Callable[[], "FilterSet"]):
        if name in FILTER_SET_REGISTRY:
            raise ValueError(f"filter set {name!r} already registered")
        FILTER_SET_REGISTRY[name] = builder
        return builder
    return deco


class FilterSet:
    """An ordered collection of `BaseFilter`s. `pass_expr()` combines them into `filter_pass`."""

    def __init__(self, filters: List[BaseFilter]):
        self.filters = list(filters)
        self._preset_name: Optional[str] = None   # stamped by `named()` for provenance readability

    # --- expressions -------------------------------------------------------
    def pass_expr(self) -> pl.Expr:
        """Combined pass/fail: each filter null->pass, AND across all. == legacy filter_pass."""
        return pl.all_horizontal(
            [f.pass_expr().fill_null(True) for f in self.filters]
        ).alias("filter_pass")

    def reasons(self) -> List[FilterReport]:
        """Ordered, flattened filter -> sub-report list. Drives first-failure attribution."""
        return [r for f in self.filters for r in f.reports()]

    # --- execution ---------------------------------------------------------
    def run(self, df, *, return_individual: bool = False, drop_failed: bool = False):
        cols = [self.pass_expr()]
        if return_individual:
            cols += [f.filter_expr() for f in self.filters]
        df = df.with_columns(cols)
        if drop_failed:
            df = df.filter(pl.col("filter_pass"))
        return df

    # --- serialisation / provenance ---------------------------------------
    @property
    def names(self) -> List[str]:
        return [f.name for f in self.filters]

    @classmethod
    def from_names(cls, names: List[str]) -> "FilterSet":
        return cls([FILTER_REGISTRY[n]() for n in names])

    @classmethod
    def named(cls, name: str) -> "FilterSet":
        try:
            builder = FILTER_SET_REGISTRY[name]
        except KeyError:
            raise KeyError(
                f"unknown filter set {name!r}; available: {sorted(FILTER_SET_REGISTRY)}"
            )
        fset = builder()
        fset._preset_name = name
        return fset

    @classmethod
    def available(cls) -> List[str]:
        return sorted(FILTER_SET_REGISTRY)


class PerLocusFilterSet:
    """An explicit map of **chains -> FilterSet**. A locus is accepted ONLY if it is listed; rows of
    any other locus (and the `_unassigned` partition) get ``filter_pass = False``.

    Construct from a dict with tuple/str keys, or an iterable of ``(chains, FilterSet)`` pairs
    (chains may be a single locus string or an iterable of them)::

        PerLocusFilterSet({("TRA", "TRB"): FilterSet.named("default_trb"),
                           ("IGH",):       FilterSet.named("default_igh")})
        PerLocusFilterSet([(["TRA", "TRB"], tcr), (["IGH"], bcr)])   # list keys via pairs

    Use :meth:`uniform` for the common "one set for every locus" case.
    `filter_pass_expr()` needs a `locus` column; `run()` assigns it first if absent.
    """

    def __init__(self, mapping):
        items = mapping.items() if isinstance(mapping, dict) else mapping
        self.groups: List[tuple] = []          # [(loci_tuple, FilterSet)]
        self.by_locus: dict = {}               # locus -> FilterSet
        for chains, fset in items:
            loci = (chains,) if isinstance(chains, str) else tuple(chains)
            if not loci:
                raise ValueError("each filter-set entry must list at least one locus")
            self.groups.append((loci, fset))
            for locus in loci:
                if locus in self.by_locus:
                    raise ValueError(f"locus {locus!r} is assigned to more than one filter set")
                self.by_locus[locus] = fset

    @classmethod
    def uniform(cls, fset: FilterSet, loci=KNOWN_LOCI) -> "PerLocusFilterSet":
        """One FilterSet applied to every locus in `loci` (default: all KNOWN_LOCI)."""
        return cls({tuple(sorted(loci)): fset})

    @property
    def accepted_loci(self) -> List[str]:
        return sorted(self.by_locus)

    def for_locus(self, locus: str) -> Optional[FilterSet]:
        """The FilterSet for `locus`, or None if that chain is not accepted."""
        return self.by_locus.get(locus)

    def filter_pass_expr(self) -> pl.Expr:
        # accepted loci -> their set's pass/fail; everything else (unlisted chains + _unassigned) -> False
        expr = None
        for loci, fset in self.groups:
            cond = pl.col("locus").is_in(list(loci))
            expr = (pl.when(cond) if expr is None else expr.when(cond)).then(fset.pass_expr())
        if expr is None:
            return pl.lit(False).alias("filter_pass")
        return expr.otherwise(pl.lit(False)).alias("filter_pass")

    def run(self, df, *, drop_failed: bool = False):
        cols = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
        if "locus" not in cols:
            df = df.with_columns(assign_locus())
        df = df.with_columns(self.filter_pass_expr())
        return df.filter(pl.col("filter_pass")) if drop_failed else df

    def provenance(self) -> dict:
        def encode(fset: FilterSet) -> dict:
            return {"preset": fset._preset_name, "filters": fset.names}
        return {locus: encode(fset) for locus, fset in self.by_locus.items()}


# --- named presets (fill in as needed) -------------------------------------

@register_filter_set("default_trb")
def _default_trb() -> FilterSet:
    return FilterSet([
        NullVFilter(),
        NullJFilter(),
        NullJunctionFilter(),
        NullJunctionAAFilter(),
        ValidJunctionAaFilter(),
        ImgtFunctionalFilter(),
    ])

@register_filter_set("no_junction_nt_trb")
def _no_junction_nt_trb() -> FilterSet:
    return FilterSet([
        NullVFilter(),
        NullJFilter(),
        NullJunctionAAFilter(),
        ValidJunctionAaFilter(),
        ImgtFunctionalFilter(),
    ])