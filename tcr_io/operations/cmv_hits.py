# use cmv ecocluster to determing number of hits in repertoire
from tcr_io.operations.base import BaseOperation, OperationResults
from tcr_io._resources import resource_path
from tcr_io.expressions import extract_genes
import polars as pl
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ECOClusterHits(BaseOperation):
    name = "ecocluster_hits"
    version = "0.2"
    description = "Counts the number of CMV ecocluster hits for each repertoire."
    supported_loci = frozenset({"TRB"})

    model_checkpoint: Optional[str] = None

    def __post_init__(self):
        path = resource_path("cmv_ecocluster.parquet") if self.model_checkpoint is None else Path(self.model_checkpoint)
        self.eco_df = pl.read_parquet(path).lazy()

    def _run(self, ds, locus: Optional[str] = None) -> OperationResults:
        
        eco_matches = []

        for repertoire_id, df in ds.iter_repertoires(progress_bar=True, progress_desc="Matching repertoires to CMV ecocluster"):
            eco_m = df.with_columns(
                *extract_genes()
            ).join(
                self.eco_df,
                on=["v_gene", "j_gene", "junction_aa"],
                how="inner"
            ).select(
                pl.lit(repertoire_id).alias("repertoire_id"),
                pl.sum("duplicate_count").alias("total_cmv_duplicates"),
                pl.median("duplicate_count").alias("median_duplicates").fill_null(0),
                pl.mean("duplicate_count").alias("mean_duplicates").fill_null(0),
                pl.len().alias("n_hits"),
                pl.n_unique("eco_id").alias("n_unique_ecoclusters"),
                pl.col("hla_cocluster").n_unique().alias("n_unique_hla_coclusters"),
            ).collect()

            eco_matches.append(eco_m)


        eco_matches = ds.repertoire_meta.select("repertoire_id", "n_clonotypes", "total_duplicates").join(
            pl.concat(eco_matches),
            on="repertoire_id",
            how="left"
        ).with_columns(
            breadth = pl.col("n_clonotypes").add(1).log() - pl.col("n_hits").add(1).log(),
        ).drop(["n_clonotypes", "total_duplicates"])
        
        return OperationResults(outputs={
            "ecocluster_hits": eco_matches,
        })
