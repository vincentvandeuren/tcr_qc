from __future__ import annotations

from .base import BaseOperation
from .._resources import resource_path
from ..expressions import extract_genes
from ..structure import Artifact, Format, Store

import polars as pl
from pathlib import Path
from abc import ABC
from dataclasses import dataclass
from scipy.spatial.distance import pdist
from itertools import combinations
import numpy as np

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..dataset import Dataset

@dataclass
class BaseHlaInferenceOperation(BaseOperation, ABC):
    name = "base_tcr2hla_inference"
    version = "0.1"
    description = "Base operation for TCR2HLA inference - does nothing"
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

    # Declared here rather than on the base: `BaseOperation.artifacts()` refuses an inherited
    # artifact, because an output's directory comes from `owner.name` — a shared declaration
    # would file both subclasses' results under `base_tcr2hla_inference/`.
    inferred_hla      = Artifact("inferred_hla.parquet", Format.PARQUET)
    inferred_hla_long = Artifact("inferred_hla_long.parquet", Format.PARQUET)

    def _run(self, ds: Dataset, out: Store) -> None:

        tcrs = self.tcrs.with_columns(
            pl.col("score").abs() # i know this is weird but this also is done in the original code. if not, it can happen that the weighted score is negative and then the logarithm is nan. If it was added just to stop that, the nans should be replaced with 0 instead, but idk.
        )

        dfs = []
        for patient_id, df in ds.iter_repertoires_by_patient(progress_bar=True, progress_desc="Inferring HLA for patients"):
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

        out(self.inferred_hla).write(
            res.sort("allele").pivot(values="model_prob", index="patient_id", on="allele")
        )
        out(self.inferred_hla_long).write(res)


@dataclass
class RepertoireHlaInference(BaseHlaInferenceOperation):
    name = "repertoire_hla_inference"
    version = "0.2"
    description = "Infers HLA types from TCR repertoires using the method from HLA2TCR. Does not group by patient"

    inferred_hla          = Artifact("inferred_hla.parquet", Format.PARQUET)
    inferred_hla_long     = Artifact("inferred_hla_long.parquet", Format.PARQUET)
    inferred_hla_distance = Artifact("inferred_hla_distance.parquet", Format.PARQUET)

    def _run(self, ds: Dataset, out: Store) -> None:

        tcrs = self.tcrs.with_columns(
            pl.col("score").abs() # i know this is weird but this also is done in the original code. if not, it can happen that the weighted score is negative and then the logarithm is nan. If it was added just to stop that, the nans should be replaced with 0 instead, but idk.
        )

        dfs = []
        for repertoire_id, df in ds.iter_repertoires(progress_bar=True, progress_desc="Inferring HLA for repertoires"):
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

        out(self.inferred_hla).write(res_wide)
        out(self.inferred_hla_long).write(res)
        out(self.inferred_hla_distance).write(pl.DataFrame({
            "repertoire_id": rep_id+rep_id_right,
            "repertoire_id_right": rep_id_right+rep_id,
            "hla_dist": np.concatenate([rep_dist, rep_dist]),
            "hla_dist_binarized": np.concatenate([rep_dist_bin, rep_dist_bin])
        }))
