# copy code from baseline technical features model
from __future__ import annotations
from .base import BaseOperation, OperationResults
from ..expressions import _determine_reference_points
from typing import TYPE_CHECKING, Literal
import polars as pl

if TYPE_CHECKING:
    from ..dataset import TcrDataset


def moment_aggs(col: str) -> list[pl.Expr]:
    """Mean, std, skew, kurtosis, quantiles."""
    aggs = [
        pl.col(col).mean().alias(f"{col}_mean"),
        pl.col(col).std().alias(f"{col}_std"),
        pl.col(col).skew().alias(f"{col}_skew"),
        pl.col(col).kurtosis().alias(f"{col}_kurtosis"),
    ]
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        aggs.append(pl.col(col).quantile(q).alias(f"{col}_q{int(q*100)}"))
    return aggs


def determine_trim_features(df:pl.LazyFrame) -> pl.DataFrame:
    df = df.with_columns(
        _determine_reference_points(
            pl.col("junction"),
            pl.col("v_call"),
            pl.col("j_call")
        ).alias("refpoints")
    ).with_columns(
        pl.col("refpoints").struct.unnest()
    ).with_columns(
        n_v_trim = pl.col("v_end") - pl.col("v_end_trimmed"),
        n_vd_trim =  pl.col("d_begin_trimmed") - pl.col("d_begin"),
        n_vj_trim = pl.col("d_end") - pl.col("d_end_trimmed"),
        n_j_trim = pl.col("j_begin_trimmed") - pl.col("j_begin"),
        n_vd_ins = pl.col("d_begin_trimmed") - pl.min_horizontal(pl.col("v_end_trimmed"), pl.col("d_begin_trimmed")),
        n_dj_ins = pl.col("j_begin_trimmed") - pl.col("d_end_trimmed")
    ).with_columns(
        n_trims = pl.col("n_v_trim") + pl.col("n_vd_trim") + pl.col("n_vj_trim") + pl.col("n_j_trim"),
        n_ins = pl.col("n_vd_ins") + pl.col("n_dj_ins")
    ).with_columns(
        frac_untemplated = pl.col("n_ins") / pl.col("cdr3_end")
    )
    
    FEATURES = [
        "n_v_trim", "n_vd_trim", "n_vj_trim", "n_j_trim",
        "n_vd_ins", "n_dj_ins", "n_trims", "n_ins",
        "cdr3_end", "frac_untemplated",
    ]

    all_aggs = []
    for feat in FEATURES:
        all_aggs.extend(moment_aggs(feat))
        all_aggs.append(pl.col(feat).sum().alias(f"{feat}_sum"))
    
    
    df = df.select(
        all_aggs
    )

    return df

class VdjStatisticsSummary(BaseOperation):
    name = "vdj_statistics_summary"
    version = "0.1"
    description = "Generates a report summarizing inferred likely VDJ trimming and insertion metrics across repertoires."


    def _run(self, ds) -> OperationResults:
        
        vdj_stat_summ = []
        for rep_id, rep in ds.iter_repertoires(filter_pass_only=True, progress_bar=True, progress_desc="Computing VDJ trimming/insertion metrics"):
            m = determine_trim_features(rep).with_columns(pl.lit(rep_id).alias("repertoire_id")).collect()
            vdj_stat_summ.append(m)

        vdj_stat_summ = pl.concat(vdj_stat_summ)
        
        return OperationResults(outputs={
            "meta/repertoire/vdj_statistics_summary.parquet": vdj_stat_summ,
        })