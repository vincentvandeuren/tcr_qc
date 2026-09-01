from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import tempfile
import polars as pl
from .base import BaseOperation
from ..expressions import KNOWN_LOCI, extract_genes
from ..structure import Artifact, Format, Store, safe_repertoire_name
from tqdm import tqdm

@dataclass
class OverlapAnalyzer(BaseOperation):
    name = "overlap_analyzer"
    version = "0.1"
    description = "Analyzes the overlap between repertoires, calculating the number of shared clonotypes and the Spearman correlation between their frequencies."
    supported_loci = KNOWN_LOCI

    heads            = Artifact("heads", Format.PARQUET_DIR)
    overlap_table    = Artifact("overlap_table.parquet", Format.PARQUET)
    overlap_overview = Artifact("overlap_overview.parquet", Format.PARQUET)

    top_n: int = 250
    junction_col: Literal["junction", "junction_aa"] = "junction"
    threshold_sus_overlap: int = 40           # different patient, above this the overlap is suspicious
    threshold_sus_overlap_absence: int = 20   # same patient, below this the absence of overlap is suspicious

    def _run(self, ds, out: Store) -> None:

        head_dir = out(self.heads).clear()

        for repertoire_id, df in ds.iter_repertoires(progress_bar=True, progress_desc="Collecting heads"):
            repertoire_id_safe = safe_repertoire_name(repertoire_id)
            df.head(self.top_n).with_columns(
                *extract_genes()
            ).sink_parquet(head_dir / f"{repertoire_id_safe}.parquet")

        _heads_tab_tmp = tempfile.TemporaryDirectory()   # ephemeral scratch, not a recorded output
        heads_tab_dir = Path(_heads_tab_tmp.name)

        pl.scan_parquet(head_dir).select(["v_gene", "j_gene", self.junction_col, "repertoire_id"]).sink_parquet(
            pl.PartitionBy(heads_tab_dir, key=["v_gene", "j_gene"])
        )
    
        total_dirs = sum(1 for _ in heads_tab_dir.glob("*/*"))
        progress_bar = tqdm(total=total_dirs, desc="Finding shared clones")
        shared_clns_list = []
        for v in heads_tab_dir.iterdir():
            for j in v.iterdir():
                shared_clns = pl.scan_parquet(j).group_by(["v_gene", "j_gene", self.junction_col]).agg(
                    pl.len().alias("count")
                ).filter(pl.col("count") > 1).collect(engine="streaming")
                shared_clns_list.append(shared_clns)
                progress_bar.update(1)
        _heads_tab_tmp.cleanup()

        shared_clns_list = pl.concat(shared_clns_list).sort("count", descending=True).with_row_index("clone_id")
        overlap = pl.scan_parquet(
            head_dir
        ).join(
            shared_clns_list.lazy(),
            on= ["v_gene", "j_gene", self.junction_col],
            how="inner"
        )
        overlap = overlap.join(
            overlap, on="clone_id", how="inner"
        ).filter(
            pl.col("repertoire_id") < pl.col("repertoire_id_right")
        ).group_by(["repertoire_id", "repertoire_id_right"]).agg(
            pl.len().alias("n_overlap"),
            pl.corr(pl.col("duplicate_count"), pl.col("duplicate_count_right"), method="pearson").fill_nan(0).alias("pearson_corr"),
            pl.corr(pl.col("duplicate_count"), pl.col("duplicate_count_right"), method="spearman").fill_nan(0).alias("spearman_corr"),
        ).collect(engine="streaming")

        # Counts are the locus's own (which repertoires are even here); patient_id is not.
        o = (ds.repertoire_counts.select(["repertoire_id", "n_filtered_clonotypes"])
               .join(ds.repertoire_meta.select(["repertoire_id", "patient_id"]), on="repertoire_id")
               .with_columns(join=pl.lit(True)))

        overlap_table = o.join(o, on="join", how="full").filter(pl.col("repertoire_id") < pl.col("repertoire_id_right")).drop(["join", "join_right"]).join(
            overlap, on=["repertoire_id", "repertoire_id_right"], how="left"
        ).with_columns(
            pl.col("n_overlap").fill_null(0),
            pl.col("pearson_corr").fill_null(0),
            pl.col("spearman_corr").fill_null(0),
            pl.col("patient_id").eq(pl.col("patient_id_right")).alias("same_patient")
        ).sort(
            "n_overlap", descending=True
        )

        overlap_table = overlap_table.with_columns(
            flag_sus_overlap = (~pl.col("same_patient")).and_(pl.col("n_overlap")>self.threshold_sus_overlap),
            flag_sus_overlap_absence = pl.col("same_patient").and_(pl.col("n_overlap")<self.threshold_sus_overlap_absence)
        )

        out(self.overlap_table).write(overlap_table)
        out(self.overlap_overview).write(overlap_table.select(
            pl.len().alias("total_pairs"),
            pl.col("flag_sus_overlap").sum().alias("sus_overlap"),
            pl.col("flag_sus_overlap_absence").sum().alias("sus_overlap_absence")
        ).with_columns(
            n_repertoires=o.height
        ))
