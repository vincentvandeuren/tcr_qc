from .base import BaseOperation
from ..expressions import KNOWN_LOCI, extract_genes
from ..structure import Artifact, Format, Store
from dataclasses import dataclass
import polars as pl


@dataclass
class TabulateByVJ(BaseOperation):
    name = "tabulate_by_vj_gene"
    version = "0.4"
    description = "Tabulates the repertoire by V and J gene for all passing rows into a hive-partitioned (v_gene, j_gene) directory."
    supported_loci = KNOWN_LOCI

    # A directory, not a file: polars sinks the hive partitions into it and it is read back
    # with `scan()`, never `read()` — PARQUET_DIR has no reader, so that is enforced rather
    # than documented.
    # v0.3: include_key=False, so the partition columns are not duplicated in the parquet files.
    # also selected only the columns needed for the tabulation, to reduce the size of the output files.
    # v0.4: added `junction` column to the output (to compute convergence later on)
    
    tabulated = Artifact("tabulated", Format.PARQUET_DIR)

    def _run(self, ds, out: Store) -> None:
        lf = ds.clonotypes.with_columns(*extract_genes()).select(["v_gene", "j_gene", "junction_aa", "junction", "repertoire_id"])
        res = lf.sink_parquet(pl.PartitionBy(out(self.tabulated).clear(), key=["v_gene", "j_gene"], include_key=False), lazy=True)
        res.collect(engine="streaming")