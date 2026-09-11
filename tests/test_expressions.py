"""Column expressions in `tcr_io.expressions` that wrap the Rust plugin."""
import polars as pl

from tcr_io.expressions import translate, trim_junction_to_cdr3


def _translated(seqs: list[str | None]) -> list[str | None]:
    """Run `translate` over a one-column frame and hand back the output column."""
    return (
        pl.DataFrame({"junction": seqs}, schema={"junction": pl.String})
        .select(translate())
        .to_series()
        .to_list()
    )


def test_translates_in_frame_nucleotides():
    assert _translated(["TGTGCCAGCTTT"]) == ["CASF"]


def test_translation_is_case_insensitive_and_returns_uppercase():
    # `trim_junction_to_cdr3` emits lowercase, so translate must accept it.
    assert _translated(["tgtgccagcttt"]) == ["CASF"]


def test_trailing_partial_codon_is_dropped():
    # 13 nt -> 4 full codons, the trailing "A" is silently ignored.
    assert _translated(["TGTGCCAGCTTTA"]) == ["CASF"]


def test_untranslatable_codon_becomes_x():
    # "NGC" has no nucleotide index -> X; flanking codons still translate.
    assert _translated(["TGTNGCTTT"]) == ["CXF"]


def test_stop_codon_is_an_asterisk():
    assert _translated(["TGTTGATTT"]) == ["C*F"]


def test_null_input_stays_null():
    assert _translated([None]) == [None]


def test_empty_string_translates_to_empty_string():
    assert _translated([""]) == [""]


def test_output_keeps_the_input_column_name():
    df = pl.DataFrame({"junction": ["TGTGCCAGCTTT"]}).select(translate())
    assert df.columns == ["junction"]


def test_translate_accepts_an_arbitrary_column():
    df = pl.DataFrame({"cdr3_nt": ["TGTGCCAGCTTT"]}).select(
        translate(pl.col("cdr3_nt")).alias("cdr3_aa")
    )
    assert df.to_series().to_list() == ["CASF"]


def test_translating_a_trimmed_junction_recovers_the_junction_aa():
    # The pipeline case: trim the junction to CDR3, translate it back, get the AA back.
    df = pl.DataFrame({
        "junction": ["GGTGTGCCAGCTTTGG"],
        "junction_aa": ["CASF"],
    }).select(translate(trim_junction_to_cdr3()).alias("round_trip"))
    assert df.to_series().to_list() == ["CASF"]
