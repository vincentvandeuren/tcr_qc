from tcr_io.operations.base import BaseOperation
from tcr_io.structure import Artifact, Format, Store
from tcr_io._resources import resource_path

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

@dataclass
class CDR3_properties(BaseOperation):
    name = "cdr3_properties"
    version = "0.2"
    description = "Computes the average amino acid properties for the CDR3 sequences in each repertoire. The average is computed both unweighted (`cdr3_properties`) and weighted by the duplicate count of each clonotype (`duplicate_weighted_cdr3_properties`)."

    cdr3_properties = Artifact("cdr3_properties.parquet", Format.PARQUET)
    duplicate_weighted_cdr3_properties = Artifact("duplicate_weighted_cdr3_properties.parquet", Format.PARQUET)

    model_checkpoint: Optional[str] = None

    def __post_init__(self):
        path = resource_path("aa_features.parquet") if self.model_checkpoint is None else Path(self.model_checkpoint)
        self.aa_features = pl.read_parquet(path).lazy()
        self.feature_cols = self.aa_features.collect_schema().names()[1:]

    def _run(self, ds, out: Store) -> None:

        def compute_features(df):

            counts = df.select(
                pl.col("junction_aa").str.split("", literal=True).alias("aa"),
                pl.col("duplicate_count")
            ).explode("aa", empty_as_null=False).group_by("aa").agg(
                pl.len().alias("aa_count"), pl.sum("duplicate_count").alias("weighted_count")
            )

            fts  = [(pl.col("aa_count").dot(f) / pl.col("aa_count").sum()).alias(f) for f in self.feature_cols] 
            wt_fts = [(pl.col("weighted_count").dot(f) / pl.col("weighted_count").sum()).alias(f+"_weighted") for f in self.feature_cols]
            all_fts = fts + wt_fts

            feature_avgs = counts.join(self.aa_features, on="aa", how="left").select(
                *all_fts
            )

            return feature_avgs


        feature_avgs_all = ds.map_repertoires(
            compute_features, progress_bar=True,
            progress_desc="Computing CDR3 properties",
        )   # eager: per-rep collect inside the helper

        feature_avgs_all = ds.repertoire_counts.select(["repertoire_id", "n_clonotypes"]).join(feature_avgs_all, on="repertoire_id", how="left")
        cdr3_properties = feature_avgs_all.select(["repertoire_id", "n_clonotypes"] + self.feature_cols)
        duplicate_weighted_cdr3_properties = feature_avgs_all.select(["repertoire_id", "n_clonotypes"] + [f+"_weighted" for f in self.feature_cols])

        out(self.cdr3_properties).write(cdr3_properties)
        out(self.duplicate_weighted_cdr3_properties).write(duplicate_weighted_cdr3_properties)
