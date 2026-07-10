import polars as pl
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .base import BaseOperation, OperationResults
from ..filters import Filterer, get_filter_summary

@dataclass
class FilteringReport(BaseOperation):
    name = "filtering_report"
    version = "0.2"
    description = "Generates a report summarizing the reasons for filtering failures across repertoires."
    supported_loci = None

    def __post_init__(self):
        # The `Filterer` (a list of polars expressions) is not serialisable config, so it is
        # not a field: `params()` stays empty and `rerun` uses the default filterer. Making the
        # filter config serialisable is a deferred follow-up (phased plan §6).
        self.filterer = Filterer(return_individual_filters=True)

    def _run(self, ds, locus: Optional[str] = None) -> OperationResults:
        filter_summ = []
        top_invalid = defaultdict(list)
                
        for rep_id, rep in ds.iter_repertoires(filter_pass_only=False, progress_bar=True, progress_desc="Creating filtering summary"):
            filtered = self.filterer.run(
                rep.filter(pl.col("filter_pass").eq(False)) # only rerun filters on failed rows to save time
            )
            filtered_filters = filtered.select(pl.col("filters")).collect()

            filter_summ.append(
                get_filter_summary(filtered_filters).with_columns(
                    repertoire_id = pl.lit(rep_id)
                )
            )

            top_invalid["invalid_junction_aa"].append(
                filtered.filter(
                    pl.col("filters").struct.field("valid_junction_aa").eq(False)
                ).group_by("junction_aa").agg(pl.len()).sort("len", descending=True).collect()
            )

            top_invalid["invalid_v_call"].append(
                filtered.filter(
                    pl.col("filters").struct.field("v_valid_imgt").eq(False)
                ).group_by("v_call").agg(pl.len()).sort("len", descending=True).collect()
            )

            top_invalid["non_functional_v_call"].append(
                filtered.filter(
                    pl.col("filters").struct.field("v_valid_imgt").eq(True)
                    & pl.col("filters").struct.field("v_func_imgt").eq(False)
                ).group_by("v_call").agg(pl.len()).sort("len", descending=True).collect()
            )

            top_invalid["invalid_j_call"].append(
                filtered.filter(
                    pl.col("filters").struct.field("j_valid_imgt").eq(False)
                ).group_by("j_call").agg(pl.len()).sort("len", descending=True).collect()
            )

            top_invalid["non_functional_j_call"].append(
                filtered.filter(
                    pl.col("filters").struct.field("j_valid_imgt").eq(True)
                    & pl.col("filters").struct.field("j_func_imgt").eq(False)
                ).group_by("j_call").agg(pl.len()).sort("len", descending=True).collect()
            )

        filter_summ = pl.concat(filter_summ).pivot(
            on = "filter",
            values = "exclusions",
            index="repertoire_id"   
        )

        top_invalid_res = dict()
        for k,reps in top_invalid.items():
            df = pl.concat(reps)
            top_invalid_res[k] = df.group_by(df.columns[0]).agg(pl.col("len").sum()).sort("len", descending=True).rename({"len":"count"})
        
        return OperationResults(outputs={
            "filter_summary": filter_summ,
            **{f"filter_top_{reason}": df for reason, df in top_invalid_res.items()}
        })