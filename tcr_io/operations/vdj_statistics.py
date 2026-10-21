# copy code from baseline technical features model
from __future__ import annotations
from .base import BaseOperation
from ..expressions import KNOWN_LOCI
from ..structure import Artifact, Format, Store
from ..expressions import _determine_reference_points
from dataclasses import dataclass
import polars as pl


FEATURES = [
    "n_v_trim", "n_vd_trim", "n_vj_trim", "n_j_trim",
    "n_v_p", "n_d5_p", "n_d3_p", "n_j_p",
    "n_vd_ins", "n_dj_ins", "n_vj_ins", "n_trims", "n_p", "n_ins",
    "cdr3_end", "frac_untemplated",
]
# Raw distributions are saved for the integer features; frac_untemplated is a ratio, only summarised.
COUNT_FEATURES = [f for f in FEATURES if f != "frac_untemplated"]

COUNTS_SCHEMA = pl.Schema({
    "repertoire_id": pl.String, "feature": pl.String, "value": pl.Int32,
    "n_clones": pl.UInt32, "n_reads": pl.Int64,
})
D_COUNTS_SCHEMA = pl.Schema({
    "repertoire_id": pl.String, "d_call": pl.String, "germline_len": pl.Int32,
    "d5_del": pl.Int32, "d3_del": pl.Int32, "d5_p": pl.Int32, "d3_p": pl.Int32,
    "n_clones": pl.UInt32, "n_reads": pl.Int64,
})


def moment_aggs(col: str) -> list[pl.Expr]:
    """Mean, std, skew, kurtosis."""
    aggs = [
        pl.col(col).mean().alias(f"{col}_mean"),
        pl.col(col).std().alias(f"{col}_std"),
        pl.col(col).skew().alias(f"{col}_skew"),
        pl.col(col).kurtosis().alias(f"{col}_kurtosis"),
    ]
    return aggs


def clone_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Per-clone reference points and V(D)J features, next to the input columns."""
    # Trimmed points include P-nucleotides, so they can lie past the germline points: germline minus
    # trimmed is deletions when positive and P-nucleotides when negative. Int32 keeps those
    # differences from wrapping around. D fields are null without a D.
    refpoint_fields = [
        "v_end_trimmed", "d_begin_trimmed", "d_end_trimmed", "j_begin_trimmed",
        "v_end", "d_begin", "d_end", "j_begin", "cdr3_end",
    ]
    has_points = pl.col("v_end_trimmed").is_not_null()
    return df.with_columns(
        _determine_reference_points(
            pl.col("junction"),
            pl.col("v_call"),
            pl.col("j_call")
        ).alias("refpoints")
    ).with_columns(
        pl.col("refpoints").struct.unnest()
    ).with_columns(
        pl.col(refpoint_fields).cast(pl.Int32)
    ).with_columns(
        n_v_trim = (pl.col("v_end") - pl.col("v_end_trimmed")).clip(lower_bound=0),
        n_v_p = (pl.col("v_end_trimmed") - pl.col("v_end")).clip(lower_bound=0),
        n_vd_trim = (pl.col("d_begin_trimmed") - pl.col("d_begin")).clip(lower_bound=0),
        n_d5_p = (pl.col("d_begin") - pl.col("d_begin_trimmed")).clip(lower_bound=0),
        n_vj_trim = (pl.col("d_end") - pl.col("d_end_trimmed")).clip(lower_bound=0),
        n_d3_p = (pl.col("d_end_trimmed") - pl.col("d_end")).clip(lower_bound=0),
        n_j_trim = (pl.col("j_begin_trimmed") - pl.col("j_begin")).clip(lower_bound=0),
        n_j_p = (pl.col("j_begin") - pl.col("j_begin_trimmed")).clip(lower_bound=0),
        n_vd_ins = pl.col("d_begin_trimmed") - pl.col("v_end_trimmed"),
        n_dj_ins = pl.col("j_begin_trimmed") - pl.col("d_end_trimmed"),
        n_vj_ins = pl.when(pl.col("d_begin_trimmed").is_null())
            .then(pl.col("j_begin_trimmed") - pl.col("v_end_trimmed")),
        # germline D bases in the match, P-nucleotides excluded
        germline_len = pl.min_horizontal("d_end", "d_end_trimmed") - pl.max_horizontal("d_begin", "d_begin_trimmed"),
    ).with_columns(
        # sum_horizontal skips nulls (no D), so rows without points must be kept null explicitly
        n_trims = pl.when(has_points).then(pl.sum_horizontal("n_v_trim", "n_vd_trim", "n_vj_trim", "n_j_trim")),
        n_p = pl.when(has_points).then(pl.sum_horizontal("n_v_p", "n_d5_p", "n_d3_p", "n_j_p")),
        n_ins = pl.when(has_points).then(pl.sum_horizontal("n_vd_ins", "n_dj_ins", "n_vj_ins")),
    ).with_columns(
        frac_untemplated = pl.col("n_ins") / pl.col("cdr3_end")
    )


def summary_aggs() -> list[pl.Expr]:
    """Moments and sum of every feature."""
    all_aggs = []
    for feat in FEATURES:
        all_aggs.extend(moment_aggs(feat))
        all_aggs.append(pl.col(feat).sum().alias(f"{feat}_sum"))
    return all_aggs


def determine_trim_features(df: pl.LazyFrame) -> pl.LazyFrame:
    return clone_features(df).select(summary_aggs())


def feature_counts(clones: pl.DataFrame) -> pl.DataFrame:
    """Clones and reads per value of each integer feature. Null values (D features without a D,
    clones without reference points) are left out."""
    return (
        clones.unpivot(on=COUNT_FEATURES, index="duplicate_count", variable_name="feature", value_name="value")
        .drop_nulls("value")
        .group_by("feature", "value")
        .agg(n_clones=pl.len(), n_reads=pl.col("duplicate_count").sum())
        .sort("feature", "value")
    )


def d_counts(clones: pl.DataFrame) -> pl.DataFrame:
    """Clones and reads per D call, germline D length, and deletions and P-nucleotides at each D
    end. Clones without a D form one row of nulls; clones without reference points are left out."""
    return (
        clones.filter(pl.col("v_end_trimmed").is_not_null())
        .group_by(
            "d_call", "germline_len",
            pl.col("n_vd_trim").alias("d5_del"), pl.col("n_vj_trim").alias("d3_del"),
            pl.col("n_d5_p").alias("d5_p"), pl.col("n_d3_p").alias("d3_p"),
        )
        .agg(n_clones=pl.len(), n_reads=pl.col("duplicate_count").sum())
        .sort("d_call", "germline_len", "d5_del", "d3_del", "d5_p", "d3_p")
    )


@dataclass
class VdjStatisticsSummary(BaseOperation):
    name = "vdj_statistics_summary"
    version = "0.4"     # v0.4: removed quantiles; raw per-value counts (vdj_statistics_counts) and D call counts (vdj_statistics_d_counts)
    # v0.3: V/J/D ends by score or exact match with P-nucleotides; deletions, P and insertions split (n_*_p, n_vj_ins); D features null without a D
    # v0.2: TRB D-features changed — the Rust D-index no longer pools TRD D-genes into TRB
    description = "Generates a report summarizing inferred likely VDJ trimming and insertion metrics across repertoires."
    supported_loci = KNOWN_LOCI

    summary = Artifact("summary.parquet", Format.PARQUET)
    counts = Artifact("counts.parquet", Format.PARQUET, schema=COUNTS_SCHEMA)
    d_counts = Artifact("d_counts.parquet", Format.PARQUET, schema=D_COUNTS_SCHEMA)

    def _run(self, ds, out: Store) -> None:
        summaries, counts, d_tables = [], [], []
        for repertoire_id, rep in ds.iter_repertoires(
            passing_only=True, progress_bar=True, progress_desc="Computing VDJ trimming/insertion metrics",
        ):
            # one pass per repertoire: the clone table is small, so the three outputs are computed in memory
            clones = clone_features(rep).collect(engine="streaming")
            tag = pl.lit(repertoire_id).alias("repertoire_id")
            summaries.append(clones.select(summary_aggs()).with_columns(tag))
            counts.append(feature_counts(clones).with_columns(tag))
            d_tables.append(d_counts(clones).with_columns(tag))
        out(self.summary).write(pl.concat(summaries))
        out(self.counts).write(pl.concat(counts))
        out(self.d_counts).write(pl.concat(d_tables))
