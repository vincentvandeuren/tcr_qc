import polars as pl
from tcr_io import determine_reference_points, trim_nucleotide_to_cdr3
from triassic.constants.preprocessing import gene_map, IMGT, adaptive_to_imgt_human


adaptive_to_imgt_human_new = {}
for g,m in adaptive_to_imgt_human.items():
    if "-X" in g:
        adaptive_to_imgt_human_new[g.replace("-X", "")] = m
    adaptive_to_imgt_human_new[g] = m


test_df = (
    pl.scan_csv("/data/datasets/public/immunecode_2020_covid_adaptive/data/raw/ImmuneCODE-Review-002/KH*.tsv", separator="\t", null_values=["na", "unknown", "no data", "unresolved"])
    .head(10_000_000)
    .rename({
        "templates":"duplicate_count",
        "amino_acid":"junction_aa",
    }
    ).select([
        "rearrangement", "v_resolved", "j_resolved", "junction_aa", "duplicate_count"
    ]).collect(engine="streaming")
)

test_df = test_df.with_columns(
    junction = trim_nucleotide_to_cdr3(pl.col("rearrangement"), pl.col("junction_aa")),
    v_call = pl.col("v_resolved").str.split("*").list.get(0).replace_strict(adaptive_to_imgt_human_new, default=None),
    j_call = pl.col("j_resolved").str.split("*").list.get(0).replace_strict(adaptive_to_imgt_human_new, default=None)
)

def time_reference_points():
    test_df.with_columns(
            reference_points = determine_reference_points(pl.col("junction"), pl.col("v_call"), pl.col("j_call"))
    )

import timeit
times = timeit.repeat("time_reference_points()", globals=globals(), number=1, repeat=5)
print(f"best: {min(times):.3f}s, all: {times}")
