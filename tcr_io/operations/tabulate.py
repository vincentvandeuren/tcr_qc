from .base import BaseOperation
from ..expressions import KNOWN_LOCI, extract_genes
from ..structure import Artifact, Format, Store
from dataclasses import dataclass
import polars as pl


@dataclass
class TabulateByVJ(BaseOperation):
    name = "tabulate_by_vj_gene"
    version = "0.2"
    description = "Tabulates the repertoire by V and J gene for all passing rows into a hive-partitioned (v_gene, j_gene) directory."
    supported_loci = KNOWN_LOCI

    # A directory, not a file: polars sinks the hive partitions into it and it is read back
    # with `scan()`, never `read()` — PARQUET_DIR has no reader, so that is enforced rather
    # than documented.
    tabulated = Artifact("tabulated", Format.PARQUET_DIR)

    def _run(self, ds, out: Store) -> None:
        # `ds` is locus-bound by the runner, so `clonotypes` is this locus's passing rows.
        lf = ds.clonotypes.with_columns(*extract_genes())
        # clear() before the sink: a partition left behind by a previous run would otherwise
        # be scanned back as if this run had produced it.
        lf.sink_parquet(pl.PartitionBy(out(self.tabulated).clear(), key=["v_gene", "j_gene"]))
