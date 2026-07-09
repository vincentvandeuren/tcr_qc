# copy code from baseline technical features model
from .base import BaseOperation, OperationResults
from ..expressions import _determine_reference_points, extract_genes
from typing import TYPE_CHECKING, Literal
import polars as pl
from itertools import product


V_GENES = ['TRBV2', 'TRBV3-1', 'TRBV4-1', 'TRBV4-2', 'TRBV4-3', 'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 'TRBV5-6', 'TRBV5-8', 'TRBV6-1', 'TRBV6-3', 'TRBV6-4', 'TRBV6-5', 'TRBV6-6', 'TRBV6-8', 'TRBV6-9', 'TRBV7-2', 'TRBV7-3', 'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9', 'TRBV10-2', 'TRBV10-3', 'TRBV11-1', 'TRBV11-2', 'TRBV11-3', 'TRBV12-3', 'TRBV12-4', 'TRBV12-5', 'TRBV13', 'TRBV14', 'TRBV15', 'TRBV16', 'TRBV18', 'TRBV19', 'TRBV20-1', 'TRBV24-1', 'TRBV25-1', 'TRBV27', 'TRBV28', 'TRBV29-1', 'TRBV30']
J_GENES = ['TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3', 'TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6', 'TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-5', 'TRBJ2-6', 'TRBJ2-7']
VJ_COLS = [f"{v}+{j}" for v, j in product(V_GENES, J_GENES)]



class GeneCountsSummary(BaseOperation):
    name = "gene_counts_summary"
    version = "0.3"
    description = "Counts the V and J gene usage frequencies, and computes the surprise of their combinations."


    def _run(self, ds) -> OperationResults:
        
        vj_table = []

        for repertoire_id, df in ds.iter_repertoires(progress_bar = True, progress_desc="Counting v/j statistics"):
            vj = df.with_columns(
                *extract_genes()
            ).group_by(["v_gene", "j_gene"]).agg(pl.len().alias("count")).with_columns(
                pl.lit(repertoire_id).alias("repertoire_id")
            ).collect()
            vj_table.append(vj)

        vj_table = pl.concat(vj_table).filter(
            pl.col("v_gene").is_in(V_GENES) & pl.col("j_gene").is_in(J_GENES)
        )

        totals = vj_table.group_by("repertoire_id").agg(pl.sum("count").alias("repertoire_total"))


        vj_table = vj_table.join(totals, on="repertoire_id").with_columns(
            (pl.col("count") / pl.col("repertoire_total")).alias("frequency")
        )

        aggs = [
            pl.col("count").sum().alias("count"),
            pl.col("frequency").sum().alias("frequency")
        ]
        v_long = vj_table.group_by(["repertoire_id", "v_gene"]).agg(aggs)
        j_long = vj_table.group_by(["repertoire_id", "j_gene"]).agg(aggs)

        vj_long = v_long.rename({"count": "count_v", "frequency": "frequency_v"}).join(
            j_long.rename({"count":"count_j", "frequency": "frequency_j"}), on="repertoire_id"
        ).join(
            vj_table.rename({"count": "count_vj", "frequency": "frequency_vj"}), on = ["repertoire_id", "v_gene", "j_gene"]
        ).with_columns(
            v_freq_p = pl.col("count_v").add(1) / pl.col("repertoire_total").add(1),
            j_freq_p = pl.col("count_j").add(1) / pl.col("repertoire_total").add(1),
            vj_freq = pl.col("count_vj").add(1) / pl.col("repertoire_total").add(1),
        ).with_columns(
            expected_vj_freq = pl.col("v_freq_p") * pl.col("j_freq_p")
        ).with_columns(
            vj_bias = pl.col("vj_freq").log() - pl.col("expected_vj_freq").log(),
            vj_comb = pl.col("v_gene").add("+").add(pl.col("j_gene"))
        ).select(["repertoire_id", "vj_comb", "vj_bias", "vj_freq"])
        
        v_tab = v_long.pivot(
            index="repertoire_id",
            on="v_gene",
            on_columns = V_GENES,
            values="frequency",
        ).fill_null(0)

        j_tab = j_long.pivot(
            index="repertoire_id",
            on="j_gene",
            on_columns = J_GENES,
            values="frequency",
        ).fill_null(0) 

        vj_surprise_tab = vj_long.pivot(
            index="repertoire_id",
            on="vj_comb",
            on_columns = VJ_COLS,
            values="vj_bias",
        ).select(
            pl.col("repertoire_id"),
            pl.col(float).fill_null(0).name.prefix("bias_")
        )

        reps = ds.repertoire_meta.select(pl.col("repertoire_id"))
        all_broad = reps.join(v_tab, on="repertoire_id", how="left").join(j_tab, on="repertoire_id", how="left").join(vj_surprise_tab, on="repertoire_id", how="left")


        return OperationResults(outputs={
            "meta/repertoire/gene_counts.parquet": all_broad,
            "meta/repertoire/v_counts.parquet": v_tab,
            "meta/repertoire/j_counts.parquet": j_tab,
            "meta/repertoire/vj_bias.parquet": vj_surprise_tab,
            "meta/repertoire/vj_long.parquet": vj_long
        })