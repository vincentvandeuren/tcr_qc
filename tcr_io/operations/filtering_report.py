"""`FilteringReport` — why rows were excluded, per locus.

The reason a row was dropped is written onto the row at ingestion (`filter_reason`), so this
op is a group-by over a stored column. It used to be a *reconstruction*: rebuild each locus's
FilterSet from the manifest's ingest provenance, re-evaluate every filter's mask against the
current code and the current IMGT reference, and attribute each failed row to the first mask
that matched. That answered "why would this row fail today", not "why was it dropped" — the
two diverge the moment a filter definition or the reference changes, which is what the
`unattributed` catch-all existed to make visible. Now the column is the answer and the
registries are consulted only to *describe* the names it holds.
"""
import polars as pl
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from .base import BaseOperation
from ..expressions import UNASSIGNED, KNOWN_LOCI
from ..filters import FilterReport, reason_index
from ..structure import Artifact, Format, Store


# A reason on disk that this library version does not define — a filter that has since been
# removed, or the placeholder a migration stamps on rows whose reason it could not recover.
# Counted like any other, but there is nothing to group it by and nothing to say about it.
_UNKNOWN_DESC = ("Reason not produced by this version of tcrio: a filter that no longer "
                 "exists, or a placeholder written by a migration.")


@dataclass
class FilteringReport(BaseOperation):
    name = "filtering_report"
    version = "0.6"   # 0.6: reads the stored `filter_reason` column instead of rebuilding it
    description = "Per-locus report of why sequences were excluded: quality failures on each present locus, assignment failures on the _unassigned partition."
    # Every real locus gets a quality report; the `_unassigned` pseudo-locus gets an assignment
    # report. Both are just loci here — the runner intersects this set with what the dataset
    # actually holds, so the _unassigned report is produced when there are unassignable rows and
    # skipped when there are none. Outputs land under operations/filtering_report/<LOCUS>/.
    supported_loci = KNOWN_LOCI | {UNASSIGNED}

    filter_summary       = Artifact("filter_summary.parquet", Format.PARQUET)
    # One table per reason. Parameterised rather than one wide table because each reason groups
    # by its OWN column (v_call for a gene reason, junction_aa for a length one) — a single frame
    # would need a column per group_col and nulls everywhere else. Only reasons that actually
    # fired get a table; "no rows failed for this reason" is an absent file, not an empty one.
    filter_top           = Artifact("filter_top_{reason}.parquet", Format.PARQUET)
    filter_reason_totals = Artifact("filter_reason_totals.parquet", Format.PARQUET)
    filter_legend        = Artifact("filter_legend.parquet", Format.PARQUET)

    top_n: int = 100   # cap for each filter_top_<reason> table (full scale is in filter_reason_totals)

    def _run(self, ds, out: Store) -> None:
        known = reason_index()
        gcols = sorted({r.group_col for r in known.values()})

        summary_rows: list[tuple] = []               # (repertoire_id, reason, count)
        top: dict[str, list] = defaultdict(list)     # reason -> [per-repertoire group-count frames]

        for rep_id, rep in ds.iter_repertoires(
            passing_only=False,
            progress_bar=True, progress_desc=f"Filtering report ({ds.locus})",
        ):
            # A non-null reason IS the exclusion. One predicate covers both populations: on a
            # real locus these are the quality failures, on `_unassigned` every row is one.
            failed = (rep.filter(pl.col("filter_reason").is_not_null())
                         .select("filter_reason", *gcols)
                         .collect())
            if failed.height == 0:
                continue

            for row in failed.group_by("filter_reason").len().iter_rows(named=True):
                summary_rows.append((rep_id, row["filter_reason"], row["len"]))

            for name in failed["filter_reason"].unique():
                report = known.get(name)
                if report is None:
                    continue                          # nothing sensible to group an unknown by
                sub = failed.filter(pl.col("filter_reason") == name)
                top[name].append(sub.group_by(report.group_col).len().rename({"len": "count"}))

        # Only reasons that fired, in the registries' order (assignment, then quality, then the
        # catch-alls), with anything unrecognised appended so it cannot be silently dropped.
        observed = {name for _, name, _ in summary_rows}
        reason_names = [n for n in known if n in observed] + sorted(observed - set(known))

        # --- per-repertoire first-failure counts (wide, one 0-filled column per reason) ---
        out(self.filter_summary).write(self._summary(summary_rows, reason_names))

        # --- top offending values per reason (capped) ------------------------
        totals: list[tuple] = []
        for name in reason_names:
            frames = top.get(name)
            if not frames:                            # unknown reason: counted, not grouped
                totals.append((name, sum(c for _, n, c in summary_rows if n == name), 0))
                continue
            merged = (pl.concat(frames)
                        .group_by(known[name].group_col).agg(pl.col("count").sum())
                        .sort("count", descending=True))
            out(self.filter_top, reason=name).write(merged.head(self.top_n))
            totals.append((name, int(merged["count"].sum()), merged.height))

        # --- reason totals: true scale behind the capped tables --------------
        out(self.filter_reason_totals).write(pl.DataFrame(
            totals,
            schema={"reason": pl.Utf8, "n_failed_rows": pl.Int64, "n_distinct_values": pl.UInt32},
            orient="row",
        ))

        # --- self-documenting legend for the reasons surfaced ----------------
        out(self.filter_legend).write(_legend(
            [known.get(n) or FilterReport(n, pl.lit(True), None, _UNKNOWN_DESC)
             for n in reason_names]
        ))

    @staticmethod
    def _summary(summary_rows: list[tuple], reason_names: List[str]) -> pl.DataFrame:
        """Wide per-repertoire failure counts: one column per reason, 0-filled.

        Columns are the reasons observed on this locus, in the report's display order, so the
        frame is rectangular even though a repertoire rarely hits every reason. Only
        repertoires that failed something appear as rows — an absent repertoire failed
        nothing."""
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


def _legend(reports: List[FilterReport]) -> pl.DataFrame:
    """Name -> description -> the column its top table groups by. One row per reason, in the
    order the report presents them."""
    return pl.DataFrame({
        "reason": [r.name for r in reports],
        "description": [r.description for r in reports],
        "group_col": [r.group_col for r in reports],
    }, schema={"reason": pl.Utf8, "description": pl.Utf8, "group_col": pl.Utf8})
