from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import polars as pl
from polars.plugins import register_plugin_function

from tcr_io._internal import __version__ as __version__

if TYPE_CHECKING:
    from tcr_io.typing import IntoExprColumn

LIB = Path(__file__).parent

# https://marcogorelli.github.io/polars-plugins-tutorial/


def _determine_reference_points(junction: pl.Expr, v_call: pl.Expr, j_call: pl.Expr, organism:str="human") -> pl.Expr:
    """
    Determines the reference points in a sequence:
    struct ReferencePoints {
    - v_end_trimmed: usize,
    - d_begin_trimmed: usize,
    - d_end_trimmed: usize,
    - j_begin_trimmed: usize,
    - v_end: usize,
    - d_begin: usize,
    - d_end: usize,
    - j_begin: usize,
    - cdr3_end: usize,
    organism: human or mouse
    }
    """
    return register_plugin_function(
        args=[junction, v_call, j_call],
        plugin_path=LIB,
        function_name="determine_reference_points_polars",
        is_elementwise=True,
        kwargs={"organism": organism},
    )

def _trim_nucleotide_to_cdr3(junction:pl.Expr, junction_aa:pl.Expr) -> pl.Expr:
    """
    Trims the nucleotide sequence to the CDR3 region based on the junction amino acid sequence.
    In the case where the junction_nt length is already equal to the junction_aa length * 3, it assumes it is already trimmed.
    Will return null if the junction_aa is not found in the translated junction sequence
    The output is in lowercase.
    """
    return register_plugin_function(
        args=[junction, junction_aa],
        plugin_path=LIB,
        function_name="trim_junction_to_cdr3_polars",
        is_elementwise=True,
    )

def _map_gene_alias(gene:pl.Expr, split_on_character: str) -> pl.Expr:
    """
    Maps a gene name to its canonical name based on the GENE_ALIASES mapping.
    If the gene name is not found in the mapping, it returns the original gene name.
    """
    return register_plugin_function(
        args=[gene],
        plugin_path=LIB,
        function_name="gene_to_imgt_canonical",
        is_elementwise=True,
        kwargs={"split_on_character": split_on_character},
    )

def _is_functional_tcr(v_call:pl.Expr, j_call:pl.Expr, organism:str="human") -> pl.Expr:
    """
    """
    return register_plugin_function(
        args=[v_call, j_call],
        plugin_path=LIB,
        function_name="is_functional_tcr",
        is_elementwise=True,
        kwargs={"organism": organism},
    )