"""Artifact / Handle / Store: path resolution, codecs, schema policy, binding."""
import json

import polars as pl
import pytest

from tcr_io.structure.store import (
    Artifact, Format, Handle, OnMissing, Store, required_dirs, resolve,
)

S = pl.Schema({"a": pl.Utf8, "b": pl.Int64})


# --- Artifact declaration -------------------------------------------------------------

def test_file_artifact_must_have_an_extension():
    """`Artifact` asserts only that a file artifact *is* a file. The exact extension is not
    asserted here: two legacy artifacts hold NDJSON under a `.json` name, and that lie is
    named once in `layout._WRONG_EXT` so it dies with the v6 rename — rather than being
    tolerated silently for every artifact forever."""
    Artifact("meta/x.parquet", Format.PARQUET)
    Artifact("meta/x.json", Format.NDJSON)          # the legacy shape, allowed here
    with pytest.raises(ValueError, match="must have a file extension"):
        Artifact("meta/x", Format.PARQUET)


def test_directory_artifact_must_not_have_an_extension():
    """The old check was `template.endswith("")`, which every string passes. A directory's
    real constraint is the opposite one, so it is now stated as such."""
    Artifact("processed/locus={locus}", Format.PARQUET_DIR)
    with pytest.raises(ValueError, match="must not have a file extension"):
        Artifact("processed/x.parquet", Format.PARQUET_DIR)


def test_template_rejects_format_specs_and_bad_names():
    with pytest.raises(ValueError, match="format specs"):
        Artifact("x/{n:03d}.parquet")
    with pytest.raises(ValueError, match="not a valid parameter name"):
        Artifact("x/{0}.parquet")


def test_empty_policy_requires_a_schema():
    with pytest.raises(ValueError, match="needs a schema"):
        Artifact("x.parquet", on_missing=OnMissing.EMPTY)


def test_set_name_records_key_and_owner():
    class Op:
        counts = Artifact("counts.parquet")
    assert Op.counts.key == "counts"
    assert Op.counts.owner is Op


def test_equality_keys_on_the_template_not_the_key():
    class A:
        x = Artifact("same.parquet")

    class B:
        y = Artifact("same.parquet")
    assert A.x == B.y and A.x.key != B.y.key


# --- resolve --------------------------------------------------------------------------

def test_resolve_strict_and_glob():
    t = "p/locus={locus}/{rid}.parquet"
    assert resolve(t, {"locus": "TRB", "rid": "s1"}) == "p/locus=TRB/s1.parquet"
    assert resolve(t, {"locus": "TRB"}, mode="glob") == "p/locus=TRB/*.parquet"
    assert resolve(t, {}, mode="glob") == "p/locus=*/*.parquet"
    with pytest.raises(KeyError, match="rid"):
        resolve(t, {"locus": "TRB"})


def test_there_are_exactly_two_modes():
    """`strict` and `glob`, and an unknown mode is an error rather than a silent passthrough.

    The design specified a third, `render`, leaving unbound parameters as literal
    `{placeholders}`. Its only consumer was `render_tree`, a pretty-printer for the dataset
    directory — deleted, because `tree` and `ls -R` already print a directory, and against the
    real one rather than a rendering of the declarations. Resolving a path and formatting a
    string for display are not the same job; this pins that they do not get merged again."""
    with pytest.raises(ValueError, match="unknown resolve mode"):
        resolve("x/{locus}.parquet", {"locus": "TRB"}, mode="render")


def test_none_counts_as_unbound():
    assert resolve("x/{locus}.parquet", {"locus": None}, mode="glob") == "x/*.parquet"


# --- round trips ----------------------------------------------------------------------

def test_parquet_round_trip_selects_and_casts(tmp_path):
    art = Artifact("t.parquet", Format.PARQUET, schema=S)
    h = Store(tmp_path)(art)
    h.write(pl.DataFrame({"junk": [9], "b": [1], "a": ["x"]}))   # wrong order, extra col
    out = h.read()
    assert out.schema == S and out.to_dicts() == [{"a": "x", "b": 1}]


def test_lazy_write_streams(tmp_path):
    h = Store(tmp_path)(Artifact("t.parquet", schema=S))
    h.write(pl.LazyFrame({"a": ["x"], "b": [1]}))
    assert h.read().height == 1


def test_empty_ndjson_is_readable_only_because_the_schema_reaches_the_reader(tmp_path):
    """audit §3.1: a bare `pl.read_ndjson` on a zero-row file raises during the read, so a
    post-hoc `_cast` cannot save it. The schema must be passed *into* the reader."""
    art = Artifact("t.ndjson", Format.NDJSON, schema=S)
    h = Store(tmp_path)(art)
    h.write(pl.DataFrame(schema=S))
    with pytest.raises(pl.exceptions.ComputeError):
        pl.read_ndjson(h.path())
    assert h.read().schema == S and h.read().height == 0


def test_ndjson_accepts_records_or_a_frame(tmp_path):
    h = Store(tmp_path)(Artifact("t.ndjson", Format.NDJSON, schema=S))
    h.write([{"a": "x", "b": 1}, {"a": "y", "b": 2}])
    assert h.read().to_dicts() == [{"a": "x", "b": 1}, {"a": "y", "b": 2}]


def test_ndjson_refuses_unserialisable_values(tmp_path):
    """No `default=str`: a value that will not serialise is an error here, not a string
    that reads back as the wrong type."""
    h = Store(tmp_path)(Artifact("t.ndjson", Format.NDJSON))
    with pytest.raises(TypeError):
        h.write([{"a": object()}])


def test_json_round_trip(tmp_path):
    h = Store(tmp_path)(Artifact("m.json", Format.JSON))
    h.write({"version": 6, "filters": {"TRB": ["productive"]}})
    assert h.read()["version"] == 6


# --- update ---------------------------------------------------------------------------

def test_update_preserves_keys_this_version_does_not_know(tmp_path):
    h = Store(tmp_path)(Artifact("m.json", Format.JSON))
    h.write({"version": 5, "filters": {"TRB": []}, "written_by_the_future": 1})
    h.update(version=6)
    assert h.read() == {"version": 6, "filters": {"TRB": []}, "written_by_the_future": 1}


def test_update_creates_when_absent(tmp_path):
    h = Store(tmp_path)(Artifact("m.json", Format.JSON, on_missing=OnMissing.NONE))
    h.update(version=6)
    assert h.read() == {"version": 6}


def test_update_is_json_only(tmp_path):
    h = Store(tmp_path)(Artifact("t.parquet"))
    with pytest.raises(TypeError, match="JSON-only"):
        h.update(version=6)


# --- on_missing -----------------------------------------------------------------------

def test_on_missing_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        Store(tmp_path)(Artifact("nope.parquet")).read()


def test_on_missing_none(tmp_path):
    assert Store(tmp_path)(Artifact("nope.parquet", on_missing=OnMissing.NONE)).read() is None


def test_on_missing_empty_is_typed_so_it_can_be_joined(tmp_path):
    """audit §3.4: an empty frame with *inferred* dtypes gives Null columns, and Null
    joined to String raises SchemaError. The empty frame is built from the schema."""
    art = Artifact("nope.parquet", schema=S, on_missing=OnMissing.EMPTY)
    empty = Store(tmp_path)(art).read()
    assert empty.schema == S and empty.height == 0
    assert pl.DataFrame({"a": ["x"]}).join(empty, on="a", how="left").height == 1


def test_on_missing_applies_to_scan_too(tmp_path):
    art = Artifact("nope.parquet", schema=S, on_missing=OnMissing.EMPTY)
    assert Store(tmp_path)(art).scan().collect().schema == S


# --- requires -------------------------------------------------------------------------

def test_requires_rejects_a_side_table_without_its_join_key(tmp_path):
    art = Artifact("extra.parquet", requires=pl.Schema({"patient_id": pl.Utf8}))
    h = Store(tmp_path)(art)
    with pytest.raises(ValueError, match="required column"):
        h.write(pl.DataFrame({"age": [40]}))
    h.write(pl.DataFrame({"patient_id": ["p1"], "age": [40]}))
    assert h.read().columns == ["patient_id", "age"]      # no schema -> nothing dropped


def test_requires_rejects_a_wrong_dtype(tmp_path):
    art = Artifact("extra.parquet", requires=pl.Schema({"patient_id": pl.Utf8}))
    with pytest.raises(ValueError, match="dtype mismatch"):
        Store(tmp_path)(art).write(pl.DataFrame({"patient_id": [1]}))


# --- sets of files --------------------------------------------------------------------

def _seed_loci(tmp_path):
    art = Artifact("counts/locus={locus}/c.parquet", schema=S)
    store = Store(tmp_path)
    for locus, n in (("TRA", 1), ("TRB", 2)):
        store(art, locus=locus).write(pl.DataFrame({"a": [locus], "b": [n]}))
    return art, store


def test_scan_reads_an_unbound_handle_as_one_frame(tmp_path):
    art, store = _seed_loci(tmp_path)
    assert sorted(store(art).scan().collect()["a"]) == ["TRA", "TRB"]
    assert store(art, locus="TRB").scan().collect().to_dicts() == [{"a": "TRB", "b": 2}]


def test_read_refuses_a_set_and_names_the_alternative(tmp_path):
    art, store = _seed_loci(tmp_path)
    with pytest.raises(ValueError, match="use scan"):
        store(art).read()


def test_write_needs_every_parameter_bound(tmp_path):
    art, store = _seed_loci(tmp_path)
    with pytest.raises(KeyError, match="locus"):
        store(art).write(pl.DataFrame({"a": ["x"], "b": [1]}))


def test_glob_lists_directories_so_an_empty_partition_survives(tmp_path):
    """audit §2.2: `present_loci` globs partition DIRECTORIES. A leaf-file glob would make
    a locus with no rows written yet disappear from the dataset."""
    art = Artifact("processed/locus={locus}", Format.PARQUET_DIR, writable=False)
    store = Store(tmp_path)
    for locus in ("TRA", "TRB", "_unassigned"):
        store(art, locus=locus).path().mkdir(parents=True)
    (tmp_path / "processed/locus=TRA/x.parquet").write_bytes(b"")   # TRB stays empty
    assert {p.name for p in store(art).glob()} == {
        "locus=TRA", "locus=TRB", "locus=_unassigned"}


def test_exists_works_bound_and_unbound(tmp_path):
    art, store = _seed_loci(tmp_path)
    assert store(art).exists()
    assert store(art, locus="TRB").exists()
    assert not store(art, locus="IGH").exists()


# --- directory artifacts --------------------------------------------------------------

def test_directory_artifact_has_no_reader_or_writer(tmp_path):
    h = Store(tmp_path)(Artifact("bundle", Format.PARQUET_DIR))
    with pytest.raises(TypeError, match="no reader"):
        h.read()
    with pytest.raises(TypeError, match="no writer"):
        h.write(pl.DataFrame({"a": ["x"]}))


def test_directory_artifact_is_scannable(tmp_path):
    art = Artifact("bundle", Format.PARQUET_DIR)
    h = Store(tmp_path)(art)
    h.path().mkdir(parents=True)
    for i in (1, 2):
        pl.DataFrame({"a": [str(i)], "b": [i]}).write_parquet(h.path() / f"{i}.parquet")
    assert h.scan().collect().height == 2


def test_clear_empties_and_recreates(tmp_path):
    h = Store(tmp_path)(Artifact("bundle", Format.PARQUET_DIR))
    h.path().mkdir(parents=True)
    (h.path() / "stale.parquet").write_bytes(b"")
    h.clear()
    assert h.path().is_dir() and not list(h.path().iterdir())


def test_clear_is_for_directories_only(tmp_path):
    with pytest.raises(TypeError, match="directory artifacts"):
        Store(tmp_path)(Artifact("t.parquet")).clear()


def test_unwritable_artifact_refuses_writes(tmp_path):
    art = Artifact("p/locus={locus}/{rid}.parquet", writable=False)
    with pytest.raises(TypeError, match="not writable"):
        Store(tmp_path)(art, locus="TRB", rid="s1").write(pl.DataFrame({"a": ["x"]}))


# --- Store binding --------------------------------------------------------------------

def test_with_params_is_additive_and_does_not_mutate(tmp_path):
    base = Store(tmp_path)
    trb = base.with_params(locus="TRB")
    assert trb.bound == {"locus": "TRB"} and base.bound == {}
    assert trb.with_params(rid="s1").bound == {"locus": "TRB", "rid": "s1"}


def test_call_overrides_bound_params(tmp_path):
    art = Artifact("p/locus={locus}/c.parquet")
    store = Store(tmp_path).with_params(locus="TRB")
    assert store(art, locus="TRA").path().parent.name == "locus=TRA"


def test_with_root_rejects_an_empty_part(tmp_path):
    """A missing locus is a bug, not an optional segment: skipping it would root the pass on
    top of the operation's own directory, where its per-locus children live."""
    base = Store(tmp_path)
    assert base.with_root("operations", "fisher", "TRB").root == tmp_path / "operations/fisher/TRB"
    with pytest.raises(ValueError, match="non-empty literal"):
        base.with_root("operations", "fisher", None)


def test_with_root_rejects_a_pattern(tmp_path):
    """C7: a root segment is a literal directory name. Reading one output across every locus
    is an iteration over locus-bound datasets, not a `*` segment."""
    with pytest.raises(ValueError, match="non-empty literal"):
        Store(tmp_path).with_root("operations", "fisher", "*")


def test_with_root_keeps_bound_params(tmp_path):
    assert Store(tmp_path).with_params(locus="TRB").with_root("operations").bound == {"locus": "TRB"}


def test_store_clear_removes_the_root(tmp_path):
    store = Store(tmp_path / "sub")
    store(Artifact("t.parquet")).write(pl.DataFrame({"a": ["x"]}))
    store.clear()
    assert not (tmp_path / "sub").exists()


# --- dir / required_dirs / create_tree -------------------------------------------------

def test_dir_is_the_parent_of_a_file_and_the_directory_itself():
    root = Store("/db").root
    assert Handle(Artifact("meta/patient/patient.parquet"), root).dir() == root / "meta/patient"
    assert Handle(Artifact("processed", Format.PARQUET_DIR), root).dir() == root / "processed"


def test_dir_truncates_at_the_first_unbound_parameter():
    """A partially-bound artifact still has a knowable home: everything above the wildcard.
    That is what lets one derivation answer both "make the tree" and "is the tree there"."""
    art = Artifact("processed/{rid}/locus={locus}/{rid}.parquet", writable=False)
    root = Store("/db").root
    assert Handle(art, root).dir() == root / "processed"
    assert Handle(art, root, {"rid": "s1"}).dir() == root / "processed/s1"
    assert Handle(art, root, {"rid": "s1", "locus": "TRB"}).dir() == root / "processed/s1/locus=TRB"


def test_required_dirs_covers_only_artifacts_whose_absence_is_an_error():
    """An optional artifact gets its directory on first write. Demanding one up front would
    call a bulk dataset malformed for lacking a single-cell table."""
    arts = [
        Artifact("meta/patient/patient.parquet"),
        Artifact("meta/clone_to_cell", Format.PARQUET_DIR, on_missing=OnMissing.NONE),
    ]
    assert [d.name for d in required_dirs("/db", arts)] == ["patient"]


def test_create_tree_makes_exactly_the_required_dirs(tmp_path):
    arts = [
        Artifact("meta/patient/patient.parquet"),
        Artifact("processed/{rid}/{rid}.parquet", writable=False),
        Artifact("meta/clone_to_cell/{rid}.parquet", on_missing=OnMissing.NONE),
    ]
    Store(tmp_path).create_tree(arts)
    assert (tmp_path / "meta/patient").is_dir()
    assert (tmp_path / "processed").is_dir()          # truncated at the unbound parameter
    assert not (tmp_path / "meta/clone_to_cell").exists()
