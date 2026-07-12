import polars as pl
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from .base import ALL_LOCI, BaseOperation, OperationResults
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

# Catch-all for a `filter_pass == False` row that no current reason explains. It should never fire:
# its presence means the stored `filter_pass` and the report's rebuilt reasons disagree — i.e.
# filter definitions or the IMGT reference drifted between ingest and report. Surfaced (rather than
# silently dropped to a null column) so that drift is visible.
_UNATTRIBUTED = FilterReport(
    "unattributed", pl.lit(True), "v_call",
    "Row failed ingest filtering but no current reason matched it — indicates the stored "
    "filter_pass and the report's rebuilt filters/reference disagree (drift since ingestion).",
)


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
    version = "0.5"   # 0.5: per-locus report (+ _unassigned pseudo-locus); capped top tables
    description = "Per-locus report of why sequences were excluded: quality failures on each present locus, assignment failures on the _unassigned partition."
    # Per present locus AND the `_unassigned` pseudo-locus (added by `loci_to_run`). Each locus's
    # outputs land under operations/filtering_report/<LOCUS>/.
    supported_loci = ALL_LOCI

    top_n: int = 100   # cap for each filter_top_<reason> table (full scale is in filter_reason_totals)

    def loci_to_run(self, present: frozenset) -> List[Optional[str]]:
        # Real loci get a quality report; `_unassigned` gets an assignment report. Always emitted
        # (empty tables when clean) so the output shape is consistent across datasets.
        return sorted(present) + [UNASSIGNED]

    def _reasons_for(self, provenance: dict, locus: str) -> List[FilterReport]:
        """The ordered *quality* reason list for a locus, rebuilt from ingest provenance (falls back
        to the default preset for pre-provenance datasets). Assignment-stage reasons are excluded —
        they can't fire on an assigned locus and are reported on the _unassigned partition instead."""
        entry = provenance.get(locus)
        fset = FilterSet.from_names(entry["filters"]) if entry else FilterSet.named("default_trb")
        return [r for r in fset.reasons() if r.name not in _ASSIGNMENT_REASON_NAMES]

    def _run(self, ds, locus: Optional[str] = None) -> OperationResults:
        is_unassigned = locus == UNASSIGNED
        reasons = ASSIGNMENT_REASONS if is_unassigned else self._reasons_for(ds._manifest.filters, locus)

        reason_gcol = {r.name: r.group_col for r in reasons}
        reason_gcol[_UNATTRIBUTED.name] = _UNATTRIBUTED.group_col
        gcols = sorted({*reason_gcol.values()})

        summary_rows: list[tuple] = []               # (repertoire_id, reason, count)
        top: dict[str, list] = defaultdict(list)     # reason -> [per-repertoire group-count frames]

        for rep_id, rep in ds.iter_repertoires(
            locus=locus, filter_pass_only=False,
            progress_bar=True, progress_desc=f"Filtering report ({locus})",
        ):
            # On a real locus we explain the FAILED rows; the `_unassigned` partition is entirely
            # assignment failures, so every row there is in scope.
            sel = rep if is_unassigned else rep.filter(pl.col("filter_pass").eq(False))
            failed = (
                sel.with_columns(first_reason(reasons).fill_null(_UNATTRIBUTED.name).alias("reason"))
                .select("reason", *gcols)
                .collect()
            )
            if failed.height == 0:
                continue

            for row in failed.group_by("reason").len().iter_rows(named=True):
                summary_rows.append((rep_id, row["reason"], row["len"]))

            for name, gcol in reason_gcol.items():
                sub = failed.filter(pl.col("reason") == name)
                if sub.height:
                    top[name].append(sub.group_by(gcol).len().rename({"len": "count"}))

        # Reason/column order: this locus's reasons, plus `unattributed` only if it ever fired.
        reason_names = [r.name for r in reasons]
        if top.get(_UNATTRIBUTED.name):
            reason_names.append(_UNATTRIBUTED.name)

        outputs: dict = {}

        # --- per-repertoire first-failure counts (wide, one 0-filled column per reason) ---
        outputs["filter_summary"] = self._summary(summary_rows, reason_names)

        # --- top offending values per reason (capped; empty table when a reason never fired) ---
        totals: list[tuple] = []
        for name in reason_names:
            gcol = reason_gcol[name]
            frames = top.get(name)
            if frames:
                merged = (
                    pl.concat(frames)
                    .group_by(gcol).agg(pl.col("count").sum())
                    .sort("count", descending=True)
                )
                outputs[f"filter_top_{name}"] = merged.head(self.top_n)
                totals.append((name, int(merged["count"].sum()), merged.height))
            else:
                outputs[f"filter_top_{name}"] = pl.DataFrame(schema={gcol: pl.Utf8, "count": pl.UInt32})
                totals.append((name, 0, 0))

        # --- reason totals: true scale behind the capped tables --------------
        outputs["filter_reason_totals"] = pl.DataFrame(
            totals,
            schema={"reason": pl.Utf8, "n_failed_rows": pl.Int64, "n_distinct_values": pl.UInt32},
            orient="row",
        )

        # --- self-documenting legend for the reasons surfaced ----------------
        outputs["filter_legend"] = _legend(
            list(reasons) + ([_UNATTRIBUTED] if top.get(_UNATTRIBUTED.name) else [])
        )

        return OperationResults(outputs=outputs)

    @staticmethod
    def _summary(summary_rows: list[tuple], reason_names: List[str]) -> pl.DataFrame:
        """Wide per-repertoire first-failure counts: one column per reason, 0-filled, stable order
        (a reason with no failures still gets its column, so the schema is comparable across data)."""
        if summary_rows:
            wide = (
                pl.DataFrame(summary_rows, schema=["repertoire_id", "reason", "count"], orient="row")
                .pivot(on="reason", values="count", index="repertoire_id", aggregate_function="sum")
                .fill_null(0)
            )
        else:
            wide = pl.DataFrame(schema={"repertoire_id": pl.Utf8})
        missing = [n for n in reason_names if n not in wide.columns]
        if missing:
            wide = wide.with_columns([pl.lit(0, dtype=pl.Int64).alias(n) for n in missing])
        return wide.select(["repertoire_id", *reason_names])
