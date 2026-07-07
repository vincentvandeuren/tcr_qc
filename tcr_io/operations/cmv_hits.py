# use cmv ecocluster to determing number of hits in repertoire
from tcr_io.operations.base import BaseOperation, OperationResults
from tcr_io._resources import resource_path
import polars as pl
from pathlib import Path

class ECOClusterHits(BaseOperation):
    name = "ecocluster_hits"
    version = "0.2"
    description = "Counts the number of CMV ecocluster hits for each repertoire."

    def __init__(self, model_checkpoint: str | Path | None = None):
        if model_checkpoint is None:
            path = resource_path("cmv_ecocluster.parquet")
        else:
            path = Path(model_checkpoint)
        self.eco_df = pl.read_parquet(path).lazy()

    def _run(self, ds) -> OperationResults:
        
        eco_matches = []

        for repertoire_id, df in ds.iter_repertoires(progress_bar=True, progress_desc="Matching repertoires to CMV ecocluster"):
            eco_m = df.with_columns(
                pl.col("v_call").str.extract(r"(.*)\*\d{2}").alias("v_gene"),
                pl.col("j_call").str.extract(r"(.*)\*\d{2}").alias("j_gene"),   
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
            "meta/repertoire/ecocluster_hits.parquet": eco_matches,
        })
