"""The migration chain runs on *old* trees, so it must not read *current* symbols.

`test_migrations_import_no_live_symbols` is the regression guard: it fails the moment a
migration reaches for `..layout`, `..schema` or `..version` again. That import is what makes
a layout change silent — a stale glob matches nothing, the step no-ops, and the next step
runs on data that was never upgraded.

`test_v3_to_head_*` are the behavioural half: a hand-built v3 tree migrated to head.
"""
import ast
import json
from pathlib import Path

import polars as pl
import pytest

from tcr_io.dataset import Dataset
from tcr_io.structure.version import DATASET_VERSION

MIGRATIONS_DIR = Path(__file__).parent.parent / "tcr_io" / "structure" / "migrations"

# What a v3 processed parquet holds, and what head holds: clonotype_id was added in v5, and
# v6 replaced the boolean with the reason it stands for.
_V3_COLUMNS = ["repertoire_id", "junction", "v_call", "junction_aa", "j_call",
               "duplicate_count", "filter_pass"]
_V6_COLUMNS = ["clonotype_id"] + _V3_COLUMNS[:-1] + ["filter_reason"]


def _v3_rows(rep_id: str, n: int) -> pl.DataFrame:
    """`n` rows, the last of which was excluded — so the v6 step has both cases to map."""
    return pl.DataFrame({
        "repertoire_id":   [rep_id] * n,
        "junction":        [f"TGT{i}" for i in range(n)],
        "v_call":          ["TRBV20-1"] * n,
        "junction_aa":     [f"CAS{i}F" for i in range(n)],
        "j_call":          ["TRBJ2-7"] * n,
        "duplicate_count": list(range(1, n + 1)),
        "filter_pass":     [True] * (n - 1) + [False],
    }).cast({"duplicate_count": pl.Int64})


@pytest.fixture
def v3_tree(tmp_path) -> Path:
    """A hand-built v3-shaped dataset: flat processed parquets, single repertoire meta table."""
    db = tmp_path / "db"
    for d in ["processed_repertoires", "meta/repertoire", "meta/patient",
              "meta/publication", "operations"]:
        (db / d).mkdir(parents=True)

    for rep_id, n in [("rep1", 3), ("rep2", 2)]:
        _v3_rows(rep_id, n).write_parquet(db / "processed_repertoires" / f"{rep_id}.parquet")

    pl.DataFrame({"repertoire_id": ["rep1", "rep2"], "patient_id": ["p1", "p2"]}) \
        .write_parquet(db / "meta/repertoire/repertoire.parquet")
    pl.DataFrame({"patient_id": ["p1", "p2"]}).write_parquet(db / "meta/patient/patient.parquet")

    (db / "meta/manifest.json").write_text(json.dumps(
        {"version": 3, "tcrio_version": "0.0.0-test", "present_loci": [], "keep_me": "yes"}))
    return db


# --------------------------------------------------------------------------- the guard

def test_migrations_import_no_live_symbols():
    """A migration may import only from `.base` and the stdlib/polars — never from the
    modules that describe the *current* dataset. Pinned literals are the whole contract."""
    forbidden = {"layout", "schema", "version", "dataset"}
    offenders = []
    for path in sorted(MIGRATIONS_DIR.glob("v*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                mod = (node.module or "").split(".")[0]
                if mod in forbidden:
                    offenders.append(f"{path.name}: from {'.' * node.level}{node.module} import "
                                     f"{', '.join(a.name for a in node.names)}")
    assert not offenders, "migrations must pin literals, not import live symbols:\n  " + \
                          "\n  ".join(offenders)


def test_migrations_take_a_path_not_a_dataset(v3_tree):
    """The signature is the enforcement: given a bare path, a migration *cannot* reach for
    the accessors that describe the current schema."""
    from tcr_io.structure import migrations
    for step in migrations.REGISTRY.values():
        step.fn(v3_tree)          # would TypeError/AttributeError on a Dataset-shaped fn


# ------------------------------------------------------------------- behaviour, v3 -> head

def test_v3_to_head_relocates_into_the_locus_first_layout(v3_tree):
    Dataset.migrated(v3_tree)

    assert (v3_tree / "processed_repertoires/locus=TRB/rep1.parquet").exists()
    assert (v3_tree / "processed_repertoires/locus=TRB/rep2.parquet").exists()
    assert not list(v3_tree.glob("processed_repertoires/*.parquet"))    # flat v3 files are gone
    assert not list(v3_tree.glob("processed_repertoires/*/locus=*"))    # v4 shells too


def test_v3_to_head_backfills_clonotype_id_per_partition(v3_tree):
    Dataset.migrated(v3_tree)

    got = pl.read_parquet(v3_tree / "processed_repertoires/locus=TRB/rep1.parquet")
    assert got.columns == _V6_COLUMNS                    # clonotype_id leads, order pinned
    assert got.schema["clonotype_id"] == pl.UInt32
    assert got["clonotype_id"].to_list() == [0, 1, 2]    # numbered within the partition
    assert got["junction"].to_list() == ["TGT0", "TGT1", "TGT2"]   # rows survive, in order

    other = pl.read_parquet(v3_tree / "processed_repertoires/locus=TRB/rep2.parquet")
    assert other["clonotype_id"].to_list() == [0, 1]     # restarts per repertoire


def test_v3_to_head_maps_filter_pass_onto_a_reason(v3_tree):
    """The boolean becomes the v6 invariant `passed <=> filter_reason.is_null()`. The reason
    itself is unrecoverable for old data, so an excluded row carries the honest placeholder."""
    Dataset.migrated(v3_tree)

    got = pl.read_parquet(v3_tree / "processed_repertoires/locus=TRB/rep1.parquet")
    assert got["filter_reason"].to_list() == [None, None, "unknown"]


def test_v3_to_head_splits_repertoire_meta_by_grain(v3_tree):
    """One table per locus becomes one locus-invariant table plus per-locus counts. The v3
    fixture never had `source_files`, so the split has to complete it rather than assume it."""
    Dataset.migrated(v3_tree)

    assert not (v3_tree / "meta/repertoire/TRB.parquet").exists()

    rep = pl.read_parquet(v3_tree / "meta/repertoire/repertoire.parquet")
    assert rep.columns == ["repertoire_id", "source_files", "patient_id"]
    assert rep["repertoire_id"].to_list() == ["rep1", "rep2"]
    assert rep["source_files"].to_list() == [None, None]

    counts = pl.read_parquet(v3_tree / "meta/repertoire/locus=TRB/counts.parquet")
    assert counts.columns == ["repertoire_id", "n_clonotypes",
                              "n_filtered_clonotypes", "total_duplicates"]


def test_v3_to_head_stamps_the_manifest_and_drops_present_loci(v3_tree):
    """`present_loci` is the shard listing now, so the copy goes; everything else survives."""
    ds = Dataset.migrated(v3_tree)

    assert ds.version == DATASET_VERSION
    man = json.loads((v3_tree / "meta/manifest.json").read_text())
    assert "present_loci" not in man
    assert man["keep_me"] == "yes"
    assert ds.dispatch_loci == frozenset({"TRB"})       # read off the tree, not the manifest


def test_migrating_twice_is_a_no_op(v3_tree):
    """Each step is idempotent, so an interrupted walk resumes rather than half-fails.

    The step re-run is the HEAD one, taken from the registry rather than imported by name:
    idempotency means a step is a no-op on the tree *it* produced, not that any older step is
    a no-op on head. Re-running v6 here would legitimately break a v7 tree — v6 renames the
    ingest record to `.ndjson`, which is exactly what v7 undoes.
    """
    from tcr_io.structure.migrations import REGISTRY
    upgrade = REGISTRY[DATASET_VERSION].fn

    Dataset.migrated(v3_tree)
    before = sorted(str(p.relative_to(v3_tree)) for p in v3_tree.rglob("*") if p.is_file())
    upgrade(v3_tree)
    assert sorted(str(p.relative_to(v3_tree)) for p in v3_tree.rglob("*") if p.is_file()) == before
    assert pl.read_parquet(v3_tree / "processed_repertoires/locus=TRB/rep1.parquet") \
             ["filter_reason"].to_list() == [None, None, "unknown"]
