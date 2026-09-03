"""The artifact table: every declared path, and the invariants that keep it honest."""
from pathlib import Path

import pytest

from tcr_io.structure import layout
from tcr_io.structure import (ARTIFACTS, Artifact, Format, OnMissing, Store,
                              required_dirs, resolve)
from tcr_io.structure import schema as S


def test_artifacts_is_the_whole_table():
    """`ARTIFACTS` is the module's artifacts — nothing declared but left out of the tuple."""
    declared = {v for v in vars(layout).values() if isinstance(v, Artifact)}
    assert declared == set(ARTIFACTS)


def test_templates_are_unique():
    assert len({a.template for a in ARTIFACTS}) == len(ARTIFACTS)


def test_every_artifact_resolves_under_a_store():
    """Every parameter is bindable, and a fully-bound handle names one path under the root."""
    store = Store("/db")
    for art in ARTIFACTS:
        params = {p: "x" for p in art.params}
        assert store(art, **params).path() == Path("/db") / resolve(art.template, params)


def test_unbound_parameters_glob_rather_than_raise():
    """A partially-bound handle is a *set*: `locus` alone leaves `repertoire_id` a wildcard."""
    pattern = resolve(layout.REPERTOIRE_FILE.template, {"locus": "TRB"}, mode="glob")
    assert pattern == "processed_repertoires/locus=TRB/*.parquet"
    with pytest.raises(KeyError):
        resolve(layout.REPERTOIRE_FILE.template, {"locus": "TRB"}, mode="strict")


def test_extension_matches_format_except_the_named_legacy_pair():
    """Every artifact's extension now matches its format — the exemption set is empty.

    Two artifacts once held NDJSON under a `.json` name, kept as a closed set so it could not
    quietly grow a third member. Both are resolved: v6 renamed `publication_ids.json` to
    `.ndjson`, and v7 made `meta/generation.json` genuinely JSON — the ingest record is one
    object, never a table. The assertion stays as an empty set rather than being deleted,
    because it is the thing that stops the exemption coming back."""
    mismatched = {a.template for a in ARTIFACTS
                  if not a.format.is_dir and not a.template.endswith(a.format.ext)}
    assert mismatched == set()


def test_empty_on_missing_implies_a_schema():
    """`OnMissing.EMPTY` builds its frame *from* the schema; without one it would yield Null
    columns that raise SchemaError on the first join."""
    for art in ARTIFACTS:
        if art.on_missing is OnMissing.EMPTY:
            assert art.schema is not None, art.template


def test_sink_written_artifacts_are_not_writable():
    """The hive leaves come out of one `PartitionBy` sink at ingest; a per-file `write()`
    would produce a tree the reader's glob does not describe."""
    for art in (layout.PROCESSED_DIR, layout.LOCUS_DIR, layout.REPERTOIRE_FILE,
                layout.CLONE_TO_CELL_DIR):
        assert not art.writable, art.template


def test_side_tables_require_only_their_join_key():
    """User columns are free; the key is not."""
    for art, key in ((layout.REPERTOIRE_EXTRA, "repertoire_id"),
                     (layout.PATIENT_EXTRA, "patient_id"),
                     (layout.PUBLICATION_EXTRA, "publication_id")):
        assert art.schema is None and art.requires is not None
        assert list(art.requires) == [key]
        assert art.on_missing is OnMissing.NONE


def test_canonical_tables_carry_their_schema():
    for art, sch in ((layout.REPERTOIRE_FILE, S.REPERTOIRE),
                     (layout.REPERTOIRE_META, S.REPERTOIRE_META),
                     (layout.PATIENT_META, S.PATIENT_META),
                     (layout.PUBLICATION_IDS, S.PUBLICATION_META),
                     (layout.HLA, S.HLA_META),
                     (layout.CLONE_TO_CELL, S.CLONE_TO_CELL)):
        assert art.schema is sch, art.template


def test_required_dirs_are_the_structural_directories():
    """The tree a dataset must have, derived from `on_missing` rather than listed twice. The
    single-cell map is absent by design: a bulk dataset never writes one."""
    assert [str(d) for d in required_dirs("", ARTIFACTS)] == [
        "meta", "meta/patient", "meta/publication", "meta/repertoire", "processed_repertoires",
    ]



def test_safe_repertoire_name_flattens_path_separators():
    assert layout.safe_repertoire_name("a/b") == "a_b"
