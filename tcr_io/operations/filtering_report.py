import polars as pl
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from .base import BaseOperation, OperationResults
from ..expressions import UNASSIGNED, KNOWN_LOCI
from ..filters import FilterSet, FilterReport


_V3 = pl.col("v_call").str.slice(0, 3)
_J3 = pl.col("j_call").str.slice(0, 3)

# Assignment-stage reasons: why a row could not be placed on a locus and rode the `_unassigned`
# partition. Ordered — a row is attributed to the FIRST reason it matches (a null v_call is
# `null_v`, never `unknown_locus`). Recomputable from v_call/j_call, so nothing is persisted.
ASSIGNMENT_REASONS: List[FilterReport] = [
    FilterReport("null_v", pl.col("v_call").is_null(), "v_call",
                 "V gene call is missing."),
    FilterReport("null_j", pl.col("j_call").is_null(), "j_call",
                 "J gene call is missing."),
    FilterReport("incompatible_locus",
                 pl.col("v_call").is_not_null() & pl.col("j_call").is_not_null() & (_V3 != _J3),
                 "v_call", "V and J belong to different loci (e.g. a TRB V with a TRA J)."),
    FilterReport("unknown_locus",
                 (_V3 == _J3) & ~_V3.is_in(list(KNOWN_LOCI)),
                 "v_call", "Gene prefix is not a recognised locus."),
]

# Reasons owned by the assignment stage. A quality FilterSet may also contain these (null_v/null_j
# live in default_trb as a standalone safety net), but on an assigned locus v/j are never null, so
# they can't fire — the report drops them from the quality reason list to keep the two populations
# cleanly separated (they're reported only on the _unassigned partition).
_ASSIGNMENT_REASON_NAMES = frozenset(r.name for r in ASSIGNMENT_REASONS)

# Empty-frame schema for the _unassigned summary (map_repertoires needs it when no rows exist).
_UNASSIGNED_SCHEMA = pl.Schema({
    "repertoire_id": pl.Utf8,
    **{r.name: pl.UInt32 for r in ASSIGNMENT_REASONS},
    "n_unassigned": pl.UInt32,
})


def first_reason(reasons: List[FilterReport]) -> pl.Expr:
    """A single `reason` column = the first reason (in order) whose mask is True on the row."""
    return pl.coalesce([
        pl.when(r.mask.fill_null(False)).then(pl.lit(r.name)) for r in reasons
    ]).alias("reason")


def _legend(reasons: List[FilterReport]) -> pl.DataFrame:
    # A reason (e.g. null_v) can appear in both the quality set and ASSIGNMENT_REASONS; the legend
    # lists each distinct reason once (first occurrence wins).
    seen: dict[str, FilterReport] = {}
    for r in reasons:
        seen.setdefault(r.name, r)
    uniq = list(seen.values())
    return pl.DataFrame({
        "reason": [r.name for r in uniq],
        "description": [r.description for r in uniq],
        "group_col": [r.group_col for r in uniq],
    })


@dataclass
class FilteringReport(BaseOperation):
    name = "filtering_report"
    version = "0.3"
    description = "Summarises why sequences were excluded: assignment failures (unassigned) and per-locus quality failures."
    supported_loci = None   # locus-agnostic: this op loops over present loci itself

    def _reasons_for(self, provenance: dict, locus: str) -> List[FilterReport]:
        """The ordered *quality* reason list for a locus, rebuilt from ingest provenance (falls back
        to the default preset for pre-provenance datasets). Assignment-stage reasons are excluded —
        they can't fire on an assigned locus and are reported on the _unassigned partition instead."""
        entry = provenance.get(locus)
        fset = FilterSet.from_names(entry["filters"]) if entry else FilterSet.named("default_trb")
        return [r for r in fset.reasons() if r.name not in _ASSIGNMENT_REASON_NAMES]

    def _summarise_unassigned(self, rep: pl.LazyFrame) -> pl.LazyFrame:
        attributed = rep.with_columns(first_reason(ASSIGNMENT_REASONS))
        return attributed.select(
            *[(pl.col("reason") == r.name).sum().cast(pl.UInt32).alias(r.name)
              for r in ASSIGNMENT_REASONS],
            pl.len().cast(pl.UInt32).alias("n_unassigned"),
        )

    def _run(self, ds, locus: Optional[str] = None) -> OperationResults:
        provenance = ds._manifest.filters
        quality_rows: list[tuple] = []               # (repertoire_id, reason, count)
        top: dict[str, list] = defaultdict(list)     # reason -> [per-(rep,locus) group counts]
        reason_meta: dict[str, FilterReport] = {}    # reason name -> FilterReport (legend/group_col)

        for loc in sorted(ds._present_loci()):
            reasons = self._reasons_for(provenance, loc)
            for r in reasons:
                reason_meta.setdefault(r.name, r)
            gcols = list({r.group_col for r in reasons})

            for rep_id, rep in ds.iter_repertoires(
                locus=loc, filter_pass_only=False,
                progress_bar=True, progress_desc=f"Filtering report ({loc})",
            ):
                failed = (
                    rep.filter(pl.col("filter_pass").eq(False))
                    .with_columns(first_reason(reasons))
                    .select("reason", *gcols)
                    .collect()
                )
                if failed.height == 0:
                    continue

                for row in failed.group_by("reason").len().iter_rows(named=True):
                    quality_rows.append((rep_id, row["reason"], row["len"]))

                for r in reasons:
                    sub = failed.filter(pl.col("reason") == r.name)
                    if sub.height:
                        top[r.name].append(
                            sub.group_by(r.group_col).len().rename({"len": "count"})
                        )

        outputs: dict = {}

        # --- per-repertoire first-failure exclusion counts (wide) ---------
        if quality_rows:
            summary = (
                pl.DataFrame(quality_rows, schema=["repertoire_id", "reason", "count"], orient="row")
                .pivot(on="reason", values="count", index="repertoire_id", aggregate_function="sum")
                .fill_null(0)
            )
        else:
            summary = pl.DataFrame(schema={"repertoire_id": pl.Utf8})
        outputs["filter_summary"] = summary

        # --- top offending values per reason ------------------------------
        for reason_name, frames in top.items():
            gcol = reason_meta[reason_name].group_col
            outputs[f"filter_top_{reason_name}"] = (
                pl.concat(frames)
                .group_by(gcol).agg(pl.col("count").sum())
                .sort("count", descending=True)
            )

        # --- assignment-stage summary on the _unassigned partition --------
        outputs["unassigned_summary"] = ds.map_repertoires(
            self._summarise_unassigned,
            locus=UNASSIGNED, filter_pass_only=False, schema=_UNASSIGNED_SCHEMA,
            progress_desc="Summarising unassigned-locus rows",
        )

        # --- self-documenting legend for every reason surfaced ------------
        outputs["filter_legend"] = _legend(
            list(reason_meta.values()) + ASSIGNMENT_REASONS
        )

        return OperationResults(outputs=outputs)
