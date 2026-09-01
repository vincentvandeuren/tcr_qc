import polars as pl
from typing import Dict, List, Literal, Optional, Union

from ._rust_expressions import _map_gene_alias, _trim_nucleotide_to_cdr3, _is_functional_tcr, _determine_reference_points

# The receptor loci tcrio understands. Phase 1 ships the TR chains; the IG chains are
# recognised here so BCR data (Phase 2) partitions correctly the moment the reference
# supports it. A 3-char gene-call prefix outside this set is treated as "no known locus".
KNOWN_LOCI: frozenset = frozenset({"TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"})

# Sentinel locus value for rows whose locus can't be derived (null/invalid v_call or j_call, or a
# prefix outside KNOWN_LOCI). It is a real `locus` value so `pl.PartitionBy` writes those rows to a
# `locus=_unassigned/` partition (instead of polars' __HIVE_DEFAULT_PARTITION__), kept for QC. The
# leading underscore marks it "not a real locus": `present_loci` excludes it (`dispatch_loci`,
# which is what op fan-out runs over, does not — the QC bucket gets a filtering report).
UNASSIGNED: str = "_unassigned"

def get_filename(
    path_col: pl.Expr = pl.col("path")
) -> pl.Expr:
    return path_col.str.extract(r"([^/]+)$").alias("filename")

def trim_junction_to_cdr3(
        junction= pl.col("junction"),
        junction_aa= pl.col("junction_aa"),
        ) -> pl.Expr:
    return _trim_nucleotide_to_cdr3(junction, junction_aa)

def to_imgt(
        gene_col = pl.col("v_call"), 
        # invalid_handling:Literal["keep_original", "null"]="keep_original",
        split_on_character:Optional[str]=None,
        fix_dv: bool=True
        ) -> pl.Expr:
    if split_on_character is None:
        split_on_character = ""
    if fix_dv:
        gene_col = gene_col.str.replace(r"(TRAV\d+(?:-\d+)?)[\\/]?DV", "${1}/DV")

    return _map_gene_alias(gene_col, split_on_character)

def is_valid_junction_aa(
        junction_aa = pl.col("junction_aa"),
        min_len:int=5,
        max_len:int=30,
        valid_amino_acids:str="ARNDCQEGHILKMFPSTWYV",
        start:list=["C"],
        end:list=["F", "W", "YV"]
    ) -> pl.Expr:
    
    valid_aa = "".join(set(valid_amino_acids)).upper()
    start_aa = "|".join(set(start)).upper()
    end_aa = "|".join(set(end)).upper()
    filter_pat = f'^{start_aa}[{valid_aa}]{{{min_len},{max_len-2}}}(?:{end_aa})$'
    return junction_aa.str.contains(filter_pat, strict=True)

def is_not_singlet(
        duplicate_count = pl.col("duplicate_count"),
    ) -> pl.Expr:
    return duplicate_count > 1

def is_functional_tcr(
        v_call = pl.col("v_call"),
        j_call = pl.col("j_call"),
        organism: Literal["human", "mouse"] = "human"
    ) -> pl.Expr:
    return _is_functional_tcr(v_call, j_call, organism).alias("is_functional")

def extract_genes(
        v_call = pl.col("v_call"),
        j_call = pl.col("j_call")
        ):
    v = v_call.str.extract(r"(.*)\*\d{2}").alias("v_gene")
    j = j_call.str.extract(r"(.*)\*\d{2}").alias("j_gene")
    return v, j


def extract_locus(
        v_call = pl.col("v_call"),
        j_call = pl.col("j_call"),
    ) -> pl.Expr:
    v_locus = v_call.str.slice(0, 3)
    j_locus = j_call.str.slice(0, 3)
    # TRAV/DV genes (e.g. TRAV14/DV4) are shared between the TRA and TRD chains, so their
    # "TRA" prefix is ambiguous. The J gene disambiguates: these rows are TRA or TRD only,
    # taken from the J prefix (any other J prefix -> undefined -> null).
    # `\d+(?:-\d+)?` matches both TRAV14/DV4 and the hyphenated family TRAV38-2/DV8.
    is_trav_dv = v_call.str.contains(r"TRAV\d+(?:-\d+)?\/DV")
    return (
        pl.when(is_trav_dv)
        .then(pl.when(j_locus.is_in(["TRA", "TRD"])).then(j_locus).otherwise(None))
        # Locus is the 3-char prefix of the gene call (TRB, IGH, ...).
        # If V and J disagree (or either is null), the row's locus is undefined -> null.
        .when(v_locus == j_locus)
        .then(v_locus)
        .otherwise(None)
        .alias("locus")
    )


def assign_locus(
        v_call = pl.col("v_call"),
        j_call = pl.col("j_call"),
        keep_loci: frozenset = KNOWN_LOCI,
    ) -> pl.Expr:
    """The `locus` column used as the ingest partition key: `extract_locus`, with any row whose
    locus is null (V/J prefixes disagree or a call is null) or outside ``keep_loci`` mapped to the
    ``UNASSIGNED`` sentinel. Every row therefore gets a concrete partition value, so
    `pl.PartitionBy(key=["locus"])` never falls back to __HIVE_DEFAULT_PARTITION__ and stray
    prefixes never create rogue `locus=<x>/` dirs — they all funnel into `locus=_unassigned/`.
    """
    locus = extract_locus(v_call, j_call)
    return (
        pl.when(locus.is_in(list(keep_loci)))
        .then(locus)
        .otherwise(pl.lit(UNASSIGNED))
        .alias("locus")
    )