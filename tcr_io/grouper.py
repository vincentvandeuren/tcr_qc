import polars as pl
from typing import List, Literal

# Columns that get a non-`first()` aggregation when collapsing duplicates.
# Everything else present in the frame is carried through with `.first()`, so
# new columns (e.g. locus, c_call, cell_id) are never silently dropped.
DEFAULT_SUM_COLS = ["duplicate_count"]
DEFAULT_JOIN_COLS = ["file"]


class Grouper:
    """
    Collapses duplicate rows into clonotypes, mirroring the `Filterer` pattern:
    a configurable object with a single `run(df) -> df` entrypoint.

    The clonotype key is chosen by `by`; every non-key column is carried through
    (schema-aware), so subclasses only need to override the small hooks
    (`_group_keys`, `_aggregations`) to change grouping behaviour — e.g. a
    single-cell grouper that retains a `cell_id` column.
    """

    CLONOTYPE_KEYS = {
        "clonotype_nt": ["v_call", "junction", "j_call"],
        "clonotype_aa": ["v_call", "junction_aa", "j_call"],
    }

    def __init__(
            self,
            by: Literal["clonotype_nt", "clonotype_aa"] = "clonotype_nt",
            sum_cols: List[str] = DEFAULT_SUM_COLS,
            join_cols: List[str] = DEFAULT_JOIN_COLS,
            sort: bool = True,
        ):
        if by not in self.CLONOTYPE_KEYS:
            raise ValueError(f"Unknown grouping '{by}'. Expected one of {list(self.CLONOTYPE_KEYS)}.")
        self.by = by
        self.sum_cols = sum_cols
        self.join_cols = join_cols
        self.sort = sort

    def _group_keys(self) -> List[str]:
        return self.CLONOTYPE_KEYS[self.by]

    def _aggregations(self, columns: List[str]) -> List[pl.Expr]:
        keys = set(self._group_keys())
        aggs = []
        for col in columns:
            if col in keys:
                continue
            if col in self.sum_cols:
                aggs.append(pl.col(col).cast(int).fill_null(1).sum().alias(col))
            elif col in self.join_cols:
                aggs.append(pl.col(col).unique().str.join(";").alias(col))
            else:
                aggs.append(pl.col(col).first())
        return aggs

    def run(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        columns = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
        grouped = df.group_by(self._group_keys()).agg(self._aggregations(columns))
        if self.sort and "duplicate_count" in columns:
            grouped = grouped.sort("duplicate_count", descending=True)
        return grouped
