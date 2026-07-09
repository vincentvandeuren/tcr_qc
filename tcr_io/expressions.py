import polars as pl
from typing import Dict, Literal, Optional

from ._rust_expressions import _map_gene_alias, _trim_nucleotide_to_cdr3, _is_functional_tcr, _determine_reference_points

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
        split_on_character:Optional[str]=None
        ) -> pl.Expr:
    if split_on_character is None:
        split_on_character = ""
    return _map_gene_alias(gene_col, split_on_character) # todo add invalid handling

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
        j_call = pl.col("j_call")
    ) -> pl.Expr:
    v = v_call.str.slice(0, 3)
    j = j_call.str.slice(0, 3)
    # Locus is the 3-char prefix of the gene call (TRB, IGH, ...).
    # If V and J disagree (or either is null), the row's locus is undefined -> null.
    return (
        pl.when(v == j)
        .then(v)
        .otherwise(None)
        .alias("locus")
    )