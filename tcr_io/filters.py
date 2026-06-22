from .expressions import is_functional_tcr, is_valid_junction_aa
import polars as pl
from typing import List

DEFAULT_FILTERS = [
    pl.col("v_call").is_not_null().alias("not_null_v"),
    pl.col("j_call").is_not_null().alias("not_null_j"),
    pl.col("junction").is_not_null().alias("not_null_junction"),
    pl.col("junction_aa").is_not_null().alias("not_null_junction_aa"),
    is_valid_junction_aa().alias("valid_junction_aa"),
    is_functional_tcr().struct.unnest(),
]

class Filterer:
    def __init__(
            self,
            filters: List[pl.Expr]=DEFAULT_FILTERS,
            return_individual_filters: bool=False,
            drop_failed_rows : bool=False
        ):
        self.filters = filters
        self.return_individual_filters = return_individual_filters
        self.drop_failed_rows = drop_failed_rows

    def run(self, df : pl.DataFrame| pl.LazyFrame) -> pl.DataFrame|pl.LazyFrame:


        df = df.with_columns(
            pl.struct(self.filters).alias("filters")
        ).with_columns(
            pl.all_horizontal(pl.col("filters").struct.unnest().fill_null(True)).alias("filter_pass")
        )
        
        if not self.return_individual_filters:
            df = df.drop("filters")

        if self.drop_failed_rows:
            df = df.filter(pl.col("filter_pass"))

        return df

def get_filter_summary(df: pl.DataFrame) -> pl.DataFrame:
    fields = df["filters"].struct.fields
    unnested = df.select(pl.col("filters").struct.unnest())

    # First-failure exclusions
    excl_exprs = []
    for i, name in enumerate(fields):
        cond = ~pl.col(name)
        for j in range(i):
            cond = cond & pl.col(fields[j])
        excl_exprs.append(cond.sum().alias(name))

    # Total false per filter
    total_false_exprs = [(~pl.col(name)).sum().alias(name) for name in fields]

    exclusions = unnested.select(excl_exprs)
    total_false = unnested.select(total_false_exprs)

    # Reshape to long and join
    result = (
        exclusions.unpivot(variable_name="filter", value_name="exclusions")
        .join(
            total_false.unpivot(variable_name="filter", value_name="total_false"),
            on="filter",
        )
    )

    return result



