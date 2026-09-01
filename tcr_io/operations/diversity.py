import polars as pl
import numpy as np
from dataclasses import dataclass
from scipy.special import binom
from .base import BaseOperation
from ..expressions import KNOWN_LOCI
from ..structure import Artifact, Format, Store

def compute_metrics(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    All diversity metrics, single repertoire, pre-sorted desc by duplicate_count
    Mostly helped by claude, should be checked. Metrics are based on the ones in Platforma.bio (but completely reimplemented here for lazy evaluation)
    """

    c = pl.col("c")

    result = (
        lf
        .sort("duplicate_count", descending=True)
        .select(pl.col("duplicate_count").cast(pl.Float64).alias("c"))
        .select(
            pl.len().cast(pl.Float64).alias("observed"),

            (c == 1.0).sum().cast(pl.Float64).alias("_f1"),
            (c == 2.0).sum().cast(pl.Float64).alias("_f2"),

            c.sum().alias("_total"),
            (c * c.log()).sum().alias("_sum_c_ln_c"),
            c.pow(2).sum().alias("_sum_c2"),

            (
                pl.when(pl.len() > 0).then(c.cum_sum().le(c.sum() * 0.5).sum() + 1)
                .otherwise(0)
            ).cast(pl.Float64).alias("d50"),

            c.cum_sum().sum().alias("_rank_c_sum"),

            *[
                (c == float(y)).sum().cast(pl.Float64).alias(f"_nx_{y}")
                for y in range(1, 21)
            ],
        )
        .with_columns(
            (pl.col("_total").log() - pl.col("_sum_c_ln_c") / pl.col("_total"))
            .alias("shannon_wiener_index"),

            (pl.col("observed")
             + pl.col("_f1") * (pl.col("_f1") - 1) / (2.0 * (pl.col("_f2") + 1)))
            .alias("chao1"),

            (pl.col("_sum_c2") / pl.col("_total").pow(2)).alias("_sum_p2"),

            (2.0 * pl.col("_rank_c_sum") / (pl.col("observed") * pl.col("_total"))
             - (pl.col("observed") + 1.0) / pl.col("observed"))
            .alias("gini"),
        )
        .with_columns(
            pl.col("shannon_wiener_index").exp().alias("shannon_wiener"),

            (pl.when(pl.col("observed") > 1)
             .then(pl.col("shannon_wiener_index") / pl.col("observed").log())
             .otherwise(0.0))
            .alias("normalized_shannon_wiener"),

            (1.0 / pl.col("_sum_p2")).alias("inverse_simpson"),
            (1.0 - pl.col("_sum_p2")).alias("gini_simpson"),
        )
        .collect()
    )

    nx = result.select([f"_nx_{y}" for y in range(1, 21)]).row(0)
    obs = result["observed"][0]
    ef = _efron_thisted(np.array(nx), obs)

    result = (
        result
        .with_columns(pl.lit(ef).alias("efron_thisted"))
        .drop([col for col in result.columns if col.startswith("_")])
    )

    for col in result.columns:
        result = result.with_columns(pl.col(col).fill_nan(0.0).fill_null(0.0))

    return result


def _efron_thisted(nx: np.ndarray, observed: float) -> float:
    if observed == 0:
        return 0.0
    for depth in range(1, 21):
        h = np.zeros(depth)
        for y in range(1, depth + 1):
            for x in range(1, y + 1):
                coeff = binom(y - 1, x - 1)
                if x % 2 == 1:
                    h[x - 1] += coeff
                else:
                    h[x - 1] -= coeff
        S = observed + sum(h[i] * nx[i] for i in range(depth))
        if S == 0:
            return 0.0
        D = np.sqrt(sum(h[i] ** 2 * nx[i] for i in range(depth)))
        CV = D / S
        if CV >= 0.05:
            break
    return float(S) if not np.isnan(float(S)) else 0.0


@dataclass
class DiversityReport(BaseOperation):
    name = "diversity_report"
    version = "0.1"
    description = "Generates a report summarizing diversity metrics across repertoires."
    supported_loci = KNOWN_LOCI

    diversity_summary = Artifact("diversity_summary.parquet", Format.PARQUET)

    def _run(self, ds, out: Store) -> None:
        out(self.diversity_summary).write(ds.map_repertoires(
            compute_metrics, passing_only=True,
            progress_bar=True, progress_desc="Computing diversity metrics",
        ))