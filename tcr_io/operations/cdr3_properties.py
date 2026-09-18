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
    version = "0.1"
    description = "Counts the number of CMV ecocluster hits for each repertoire."

    cdr3_properties = Artifact("cdr3_properties.parquet", Format.PARQUET)

    model_checkpoint: Optional[str] = None

    def __post_init__(self):
        path = resource_path("aa_features.parquet") if self.model_checkpoint is None else Path(self.model_checkpoint)
        self.aa_features = pl.read_parquet(path).lazy()
        self.feature_cols = self.aa_features.collect_schema().names()[1:]

    def _run(self, ds, out: Store) -> None:

        def compute_features(df):
            # repertoire_id is tagged by map_repertoires; per-rep aggregate is one row.
            counts = df.select(
                pl.col("junction_aa").str.split("", literal=True).explode(empty_as_null=False).alias("aa")
            ).group_by("aa").agg(pl.len().alias("aa_count"))

            feature_avgs = counts.join(self.aa_features, on="aa", how="left").select(
                *[(pl.col("aa_count").dot(f) / pl.col("aa_count").sum()).alias(f) for f in self.feature_cols]
            )

            return feature_avgs


        feature_avgs_all = ds.map_repertoires(
            compute_features, progress_bar=True,
            progress_desc="Computing CDR3 properties",
        )   # eager: per-rep collect inside the helper

        feature_avgs_all = ds.repertoire_counts.select(["repertoire_id", "n_clonotypes"]).join(feature_avgs_all, on="repertoire_id", how="left")

        out(self.cdr3_properties).write(feature_avgs_all)
