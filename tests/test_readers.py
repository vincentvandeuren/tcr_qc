"""Reader layer: registry discovery, optional columns, and the default `_process`."""
import polars as pl

from tcr_io.readers import (
    READER_REGISTRY, ReaderFactory, AirrReader, CellrangerReader, TcrdistReader, PirdReader,
)


def test_registry_discovers_readers_including_subclasses():
    # subclass-of-subclass readers register too — no ", BaseReader" mixin needed anymore
    assert {"airr", "cellranger", "pird", "tcrdb", "mixcr_from_pogorelyy_2018"} <= set(READER_REGISTRY)
    assert PirdReader.__mro__[1] is AirrReader                 # plain single inheritance
    assert len(ReaderFactory().readers) == len(READER_REGISTRY)


def test_optional_col_map_present_absent_and_required_miss():
    airr = AirrReader()
    base = ["junction", "junction_aa", "v_call", "j_call", "count"]
    assert airr._determine_col_map(base + ["cell_id"])["cell_id"] == "cell_id"   # present -> mapped
    assert "cell_id" not in airr._determine_col_map(base)                        # absent -> skipped
    assert airr._determine_col_map(["v_call", "j_call"]) is None                 # required miss -> None


def test_optional_does_not_override_a_required_mapping():
    # Cellranger maps barcode->cell_id (required); an extra literal cell_id column must not double-map
    cm = CellrangerReader()._determine_col_map(
        ["cdr3_nt", "cdr3", "v_gene", "j_gene", "umis", "barcode", "cell_id"])
    assert cm["barcode"] == "cell_id"
    assert "cell_id" not in cm            # the optional cell_id->cell_id was skipped (guard)


def test_default_process_canonicalises_calls():
    # TcrdistReader has no _process -> inherits the base default (to_imgt + trim junction)
    lf = pl.LazyFrame({
        "v_call": ["TRBV2"], "j_call": ["TRBJ2-1"],
        "junction": ["TGTGCCAGCAGCCTG"], "junction_aa": ["CASSL"],
    })
    out = TcrdistReader()._process(lf).collect()
    assert out["v_call"][0] == "TRBV2*01"        # bare gene -> canonical allele
    assert out["j_call"][0] == "TRBJ2-1*01"
