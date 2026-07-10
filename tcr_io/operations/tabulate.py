from .base import ALL_LOCI, BaseOperation, OperationResults
from ..expressions import extract_genes
from dataclasses import dataclass
from typing import Optional
import polars as pl

@dataclass
class TabulateByVJ(BaseOperation):
    name = "tabulate_by_vj_gene"
    version = "0.2"     # output is now a Kind.UNSTRUCTURED dir read back as a Path (was hive)
    description = "Tabulates the repertoire by V and J gene for all passing rows into a hive-partitioned (v_gene, j_gene) directory."
    supported_loci = ALL_LOCI

    def _run(self, ds, locus: Optional[str] = None) -> OperationResults:
        lf = pl.scan_parquet(ds.repertoire_dir / locus).filter("filter_pass").with_columns(
            *extract_genes()
        )
        # Unstructured directory output: the op sinks the hive partitions itself into the
        # managed dir (operations/<op>/[<locus>/]tabulated/). It is recorded by the framework
        # and read back as a Path — scan it with pl.scan_parquet(path, hive_partitioning=True).
        out = ds._operation_output_dir(self, "tabulated", locus)
        lf.sink_parquet(pl.PartitionBy(out, key=["v_gene", "j_gene"]))
        return OperationResults(outputs={})
