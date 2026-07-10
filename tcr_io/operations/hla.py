from __future__ import annotations

from .base import BaseOperation, OperationResults
from .._resources import resource_path
from ..expressions import extract_genes, extract_locus

import polars as pl
from pathlib import Path
from abc import ABC
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import numpy as np

from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from ..dataset import TcrDataset

@dataclass
class BaseHlaInferenceOperation(BaseOperation, ABC):
    name = "base_tcr2hla_inference"
    version = "0.1"
    description = "Base operation for TCR2HLA inference - does nothing"
    # TRB-intrinsic model -> one locus-agnostic pass; the op filters to TRB internally and
    # its outputs carry no <LOCUS>/ segment (see operations_restructure_phased_plan §3 wrinkle).
    supported_loci = frozenset({"TRB"})   # TCR2HLA model is TRB-specific

    model_checkpoint: Optional[str] = None

    def __post_init__(self):
        if self.model_checkpoint is None:
            tcrs_path = resource_path("hla_tcrs.parquet")
            wts_path = resource_path("hla_model_weights.parquet")
        else:
            d = Path(self.model_checkpoint)
            tcrs_path = d / "hla_tcrs.parquet"
            wts_path = d / "hla_model_weights.parquet"
        self.tcrs = pl.read_parquet(tcrs_path).lazy()
        self.wts = pl.read_parquet(wts_path).lazy()


@dataclass
class HlaInference(BaseHlaInferenceOperation):
    name = "hla_inference"
    version = "0.4"
    description = "Infers HLA types from TCR repertoires using the method from HLA2TCR."


    def _run(self, ds: TcrDataset, locus: Optional[str] = None) -> OperationResults:

        tcrs = self.tcrs.with_columns(
            pl.col("score").abs() # i know this is weird but this also is done in the original code. if not, it can happen that the weighted score is negative and then the logarithm is nan. If it was added just to stop that, the nans should be replaced with 0 instead, but idk.
        )

        dfs = []
        for patient_id, df in ds.iter_repertoires_by_patient(locus=locus, progress_bar=True, progress_desc="Inferring HLA for patients"):
            df = df.filter(
                extract_locus() == "TRB"
            )
            n_tcrs = df.select(pl.len()).collect()[0,0]
            df = df.with_columns(
                *extract_genes()
            ).lazy().join(
                tcrs, on=["v_gene", "j_gene", "junction_aa"], how="inner"
            ).group_by("allele").agg(
                pl.sum("duplicate_count").alias("total_duplicate_count"),
                pl.len().alias("n_unique_hits"),
                pl.col("score").sum().alias("sum_score"),
                (pl.col("score")*pl.col("duplicate_count")).sum().add(1).log(10)
            ).join(
                self.wts.select("allele"), on="allele", how="right"
            ) .with_columns(
                pl.col("score").fill_null(0),
                pl.col("n_unique_hits").fill_null(0),
                pl.col("sum_score").fill_null(0),
                pl.lit(patient_id).alias("patient_id"),
                pl.lit(n_tcrs).add(1).log(10).alias("log10p_n_tcrs"),
            )

            dfs.append(df.collect())  
        
        res = pl.concat(dfs).join(
            self.wts.collect(), on="allele", how="left"
        ).with_columns(
            model_score = pl.col("score") * pl.col("coef_1") + pl.col("log10p_n_tcrs") * pl.col("coef_2") + pl.col("intercept")
        ).with_columns(
            model_prob = 1 / (1 + (-pl.col("model_score")).exp())
        )

        res_wide = res.sort("allele").pivot(values="model_prob", index="patient_id", on="allele")

        return OperationResults(
            outputs = {
                "inferred_hla": res_wide,
                "inferred_hla_long": res,
            }
        )


@dataclass
class RepertoireHlaInference(BaseHlaInferenceOperation):
    name = "repertoire_hla_inference"
    version = "0.2"
    description = "Infers HLA types from TCR repertoires using the method from HLA2TCR. Does not group by patient"


    def _run(self, ds: TcrDataset, locus: Optional[str] = None) -> OperationResults:

        tcrs = self.tcrs.with_columns(
            pl.col("score").abs() # i know this is weird but this also is done in the original code. if not, it can happen that the weighted score is negative and then the logarithm is nan. If it was added just to stop that, the nans should be replaced with 0 instead, but idk.
        )

        dfs = []
        for repertoire_id, df in ds.iter_repertoires(locus=locus, progress_bar=True, progress_desc="Inferring HLA for repertoires"):
            df = df.filter(
                extract_locus() == "TRB"
            )
            n_tcrs = df.select(pl.len()).collect()[0,0]
            df = df.with_columns(
                *extract_genes()
            ).lazy().join(
                tcrs, on=["v_gene", "j_gene", "junction_aa"], how="inner"
            ).group_by("allele").agg(
                pl.sum("duplicate_count").alias("total_duplicate_count"),
                pl.len().alias("n_unique_hits"),
                pl.col("score").sum().alias("sum_score"),
                (pl.col("score")*pl.col("duplicate_count")).sum().add(1).log(10)
            ).join(
                self.wts.select("allele"), on="allele", how="right"
            ) .with_columns(
                pl.col("score").fill_null(0),
                pl.col("n_unique_hits").fill_null(0),
                pl.col("sum_score").fill_null(0),
                pl.lit(repertoire_id).alias("repertoire_id"),
                pl.lit(n_tcrs).add(1).log(10).alias("log10p_n_tcrs"),
            )

            dfs.append(df.collect())  
        
        res = pl.concat(dfs).join(
            self.wts.collect(), on="allele", how="left"
        ).with_columns(
            model_score = pl.col("score") * pl.col("coef_1") + pl.col("log10p_n_tcrs") * pl.col("coef_2") + pl.col("intercept")
        ).with_columns(
            model_prob = 1 / (1 + (-pl.col("model_score")).exp())
        )

        res_wide = res.sort("allele").pivot(values="model_prob", index="repertoire_id", on="allele")

        hla_mat = res_wide.drop("repertoire_id").to_numpy()
        rep_dist = pdist(hla_mat, metric="euclidean")
        rep_dist_bin = pdist(hla_mat>0.5, metric="hamming")*hla_mat.shape[1]
        rep_id, rep_id_right = zip(*combinations(res_wide["repertoire_id"], 2))

        rep_dist_df = pl.DataFrame({
            "repertoire_id": rep_id+rep_id_right,
            "repertoire_id_right": rep_id_right+rep_id,
            "hla_dist": np.concatenate([rep_dist, rep_dist]),
            "hla_dist_binarized": np.concatenate([rep_dist_bin, rep_dist_bin])
        })


        return OperationResults(
            outputs = {
                "inferred_hla": res_wide,
                "inferred_hla_long": res,
                "inferred_hla_distance": rep_dist_df,
            }
        )


def get_hla_class_counts(ds:TcrDataset) -> pl.DataFrame:

    h = ds.get_operation_result("repertoire_hla_inference", "inferred_hla_long")

    inf = HlaInference()

    hla_summ_long = h.with_columns(
        pl.col("allele").str.extract(r"^([ABDCQPR]+)\-").alias("gene")
    ).join(
        inf.tcrs.group_by("allele").agg(pl.len().alias("n_max")).collect(), on="allele", how="left"
    ).with_columns(
        n_unique_hits = pl.col("n_unique_hits"),
        model_prob_weighted_n_unique_hits = pl.col("model_prob") * pl.col("n_unique_hits"),
        weighted_capped_n_unique_hits = pl.col("model_prob") * pl.col("n_unique_hits") * pl.col("n_max"),
        log_dups = pl.col("total_duplicate_count").add(1).log(2)
    ).group_by(["repertoire_id", "gene"]).agg(
        pl.sum("n_unique_hits").alias("total_n_unique_hits"),
        pl.sum("weighted_capped_n_unique_hits").alias("total_weighted_capped_n_unique_hits"),
        pl.sum("model_prob_weighted_n_unique_hits").alias("total_model_prob_weighted_n_unique_hits"),
        pl.sum("log_dups").alias("total_duplicate_count")
    )

    h_count = hla_summ_long.pivot(
        index="repertoire_id", on="gene", values="total_n_unique_hits"
    ).select(["repertoire_id", "A", "B", "C", "DP", "DQ", "DR"]).with_columns(
        class_i_to_ii_ratio = (pl.col("A") + pl.col("B") + pl.col("C")).add(1).log().cast(pl.Float64) - (pl.col("DP") + pl.col("DQ") + pl.col("DR")).add(1).log().cast(pl.Float64),
        class_i = (pl.col("A") + pl.col("B") + pl.col("C")),
        class_ii = (pl.col("DP") + pl.col("DQ") + pl.col("DR"))
    )

    h_wt_count = hla_summ_long.pivot(
        index="repertoire_id", on="gene", values="total_model_prob_weighted_n_unique_hits"
    ).select(["repertoire_id", "A", "B", "C", "DP", "DQ", "DR"]).with_columns(
        class_i_to_ii_ratio = (pl.col("A") + pl.col("B") + pl.col("C")).add(1).log().cast(pl.Float64) - (pl.col("DP") + pl.col("DQ") + pl.col("DR")).add(1).log().cast(pl.Float64),
        class_i = (pl.col("A") + pl.col("B") + pl.col("C")),
        class_ii = (pl.col("DP") + pl.col("DQ") + pl.col("DR"))
    )

    h_wt_capped = hla_summ_long.pivot(
        index="repertoire_id", on="gene", values="total_weighted_capped_n_unique_hits"
    ).select(["repertoire_id", "A", "B", "C", "DP", "DQ", "DR"]).with_columns(
        class_i_to_ii_ratio = (pl.col("A") + pl.col("B") + pl.col("C")).add(1).log().cast(pl.Float64) - (pl.col("DP") + pl.col("DQ") + pl.col("DR")).add(1).log().cast(pl.Float64),
        class_i = (pl.col("A") + pl.col("B") + pl.col("C")),
        class_ii = (pl.col("DP") + pl.col("DQ") + pl.col("DR"))
    )

    # h_duplicates = hla_summ_long.pivot(
    #     index="repertoire_id", on="gene", values="total_duplicate_count"
    # ).select(["repertoire_id", "A", "B", "C", "DP", "DQ", "DR"]).with_columns(
    #     class_i_to_ii_ratio = (pl.col("A") + pl.col("B") + pl.col("C")).add(1).log().cast(pl.Float64) - (pl.col("DP") + pl.col("DQ") + pl.col("DR")).add(1).log().cast(pl.Float64),
    #     class_i = (pl.col("A") + pl.col("B") + pl.col("C")),
    #     class_ii = (pl.col("DP") + pl.col("DQ") + pl.col("DR"))
    # )

    h_counts = h_count.join(h_wt_count, on="repertoire_id", how="left", suffix="_wt").join(h_wt_capped, on="repertoire_id", how="left", suffix="_wt_with_max")

    return h_counts