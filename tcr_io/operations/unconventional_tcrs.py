import math

from tcr_io.operations.base import BaseOperation, OperationResults
from tcr_io._resources import resource_path
from tcr_io.expressions import extract_genes
import polars as pl
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional

class BaseHits(BaseOperation, ABC):
    name = "base_hits"
    version = "0.0"
    description = "Counts the number of hits for each repertoire."
    result_name_short = "base"

    def __init__(self, model_checkpoint: Optional[str | Path] = None):
        if model_checkpoint is None:
            raise ValueError(
                f"{type(self).__name__} requires a model file (not bundled in the beta). "
                "Pass model_checkpoint=<path-to-parquet>. See README 'Beta limitations'."
            )
        self.query_df = self._prepare_query_df(Path(model_checkpoint))

    @abstractmethod
    def _prepare_query_df(self, model_checkpoint:Path) -> pl.LazyFrame:
        pass

    def _run(self, ds) -> OperationResults:
        """
        Clonal breadth and depth of disease-specific T-cell response.
        As defined in Snyder et al https://doi.org/10.3389/fimmu.2024.1488860

        Definitions:
            N_j = number of unique TCR clonotypes in repertoire j
            t_ij = template count (duplicate_count) for clone i in repertoire j
            M_j = sum_i(t_ij) = total templates sequenced in repertoire j
            D = set of disease-associated sequences
            g_ij = log2(1 + t_ij) = estimated clonal generations for lineage i

        Breadth (Eq 1):
            B_j = (1 / N_j) * sum_{i in D} I(t_ij > 0)

            Proportion of repertoire clonotypes matching disease set.

        Depth (Eq 2):
            D_j = sum_{i in D} g_ij - log2(M_j)

            Relative clonal expansion generations across disease-associated
            clones, normalized by total repertoire size.

        Breadth error (Eq 3):
            dB_j = B_j * sqrt(1/N_j + 1/sum_{i in D} I(t_ij > 0))

            Poisson error on counting statistics for numerator and denominator.

        Depth error (Eq 4):
            dg_ij = sqrt(t_i) / ((1 + t_i) * ln2)
            dlog2_Mj = sqrt(M_j) / (M_j * ln2)
            dD_j = sqrt(sum_{i in D} (dg_ij)^2 + (dlog2_Mj)^2)

            Propagated Poisson counting errors on template counts, added in quadrature.
        
        Evenness: (custom metric, not in Snyder et al)
        p_ij = t_ij / sum_{k in D}(t_kj)
        H_j = -sum_{i in D} p_ij * ln(p_ij)
             = ln(S) - (1/S) * sum_{i in D} t_ij * ln(t_ij)   where S = sum(t_ij)
        E_j = exp(H_j) / n_hits

        Shannon evenness (Hill q=1 / Hill q=0). Ranges 0-1.
        Low = few dominant clones. High = uniform expansion.
        """
        name = self.result_name_short

        ms = []
        for repertoire_id, df in ds.iter_repertoires(progress_bar=True, progress_desc=f"Counting {name} hits in repertoires"):
            repertoire_lf = df.with_columns(
                *extract_genes()
            )

            # N_j, M_j from full repertoire
            repo_stats = repertoire_lf.select(
                pl.len().alias("N_j"),
                pl.sum("duplicate_count").alias("M_j"),
            )

            # All hit-level aggregations in one pass
            hit_stats = repertoire_lf.join(
                self.query_df.lazy(),
                on=["v_gene", "j_gene", "junction_aa"],
                how="inner",
            ).select(
                pl.len().alias("n_hits"),
                pl.sum("duplicate_count").alias("total_dup"),
                pl.median("duplicate_count").alias("median_dup"),
                pl.mean("duplicate_count").alias("mean_dup"),
                pl.col("duplicate_count").add(1).log(2).sum().alias("g_sum"),
                (pl.col("duplicate_count").sqrt() / ((pl.col("duplicate_count") + 1) * pl.lit(2).log()))
                    .pow(2).sum().alias("dg2_sum"),
                # For Shannon entropy: sum(t_i * ln(t_i))
                (pl.col("duplicate_count").cast(pl.Float64)
                * pl.col("duplicate_count").cast(pl.Float64).log())
                    .sum().alias("t_ln_t_sum"),
            )

            # Combine + derive final metrics — single collect
            m = repo_stats.join(hit_stats, how="cross").select(
                pl.lit(repertoire_id).alias("repertoire_id"),
                pl.col("n_hits").fill_null(0).alias(f"n_{name}_hits"),
                pl.col("total_dup").fill_null(0).alias(f"total_{name}_duplicates"),
                pl.col("median_dup").fill_null(0.0).alias(f"median_{name}_duplicates"),
                pl.col("mean_dup").fill_null(0.0).alias(f"mean_{name}_duplicates"),
                # Breadth (Eq 1)
                (pl.col("n_hits") / pl.col("N_j")).fill_null(0.0).alias(f"{name}_breadth"),
                # Breadth error (Eq 3)
                pl.when(pl.col("n_hits") > 0).then(
                    (pl.col("n_hits") / pl.col("N_j"))
                    * (1.0 / pl.col("N_j") + 1.0 / pl.col("n_hits")).sqrt()
                ).otherwise(0.0).alias(f"{name}_breadth_err"),
                # Depth (Eq 2)
                pl.when(pl.col("n_hits") > 0).then(
                    pl.col("g_sum") - pl.col("M_j").cast(pl.Float64).log(2)
                ).otherwise(0.0).alias(f"{name}_depth"),
                # Depth error (Eq 4)
                pl.when(pl.col("n_hits") > 0).then(
                    (pl.col("dg2_sum")
                    + (1.0 / (pl.col("M_j").cast(pl.Float64) * pl.lit(2).log() * pl.lit(2).log()))).sqrt()
                ).otherwise(0.0).alias(f"{name}_depth_err"),
                # Evenness: exp(H) / n_hits where H = ln(S) - (1/S)*sum(t*ln(t))
                pl.when(pl.col("n_hits") > 1).then(
                    (pl.col("total_dup").cast(pl.Float64).log()
                    - pl.col("t_ln_t_sum") / pl.col("total_dup").cast(pl.Float64))
                    .exp() / pl.col("n_hits").cast(pl.Float64)
                ).when(pl.col("n_hits") == 1).then(1.0)
                .otherwise(0.0).alias(f"{name}_evenness"),
            ).collect()

            ms.append(m)

        hits_df = ds.repertoire_meta.select(["repertoire_id"]).join(
            pl.concat(ms),
            on="repertoire_id",
            how="left",
        )
        return OperationResults(outputs={
            f"meta/repertoire/{self.result_name_short}_hits.parquet": hits_df,
        })

class MaitHits(BaseHits):
    name = "mait_hits"
    version = "0.6"
    description = "Counts the number of mait hits (from unconventional tcr db) for each repertoire."
    result_name_short = "mait"

    def __init__(self, model_checkpoint: Optional[str | Path] = None):
        # The MAIT reference set is bundled; default to it, allow a path override.
        if model_checkpoint is None:
            model_checkpoint = resource_path("mait_hits.parquet")
        super().__init__(model_checkpoint)

    def _prepare_query_df(self, model_checkpoint:Path) -> pl.LazyFrame:
        return pl.read_parquet(model_checkpoint).select(["v_gene", "j_gene", "junction_aa"]).lazy()