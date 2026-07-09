from .base import BaseOperation, OperationResults
from ..expressions import extract_genes
import os
import polars as pl

class TabulateByVJ(BaseOperation):
    name = "tabulate_by_vj_gene"
    version = "0.1"
    description = "Tabulates the repertoire by V and J gene genes for all passing rows, creating a hive-partitioned table."

    def _run(self, ds) -> OperationResults:
        pl.scan_parquet(ds.repertoire_dir).filter("filter_pass").with_columns(
            *extract_genes()
        ).sink_parquet(
            pl.PartitionBy(ds.db_dir/"tabulated", key=["v_gene", "j_gene"])
        )

        return OperationResults(outputs={})
