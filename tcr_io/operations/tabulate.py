from .base import BaseOperation, OperationResults
import os
import polars as pl

class TabulateByVJ(BaseOperation):
    name = "tabulate_by_vj_gene"
    version = "0.1"
    description = "Tabulates the repertoire by V and J gene genes for all passing rows, creating a hive-partitioned table."

    def _run(self, ds) -> OperationResults:
        pl.scan_parquet(ds.repertoire_dir).filter("filter_pass").with_columns(
            v_gene = pl.col("v_call").str.extract(r"(.*)\*\d{2}"),
            j_gene = pl.col("j_call").str.extract(r"(.*)\*\d{2}")
        ).sink_parquet(
            pl.PartitionBy(ds.db_dir/"tabulated", key=["v_gene", "j_gene"])
        )

        return OperationResults(outputs={})
