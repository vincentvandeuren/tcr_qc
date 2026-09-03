"""Version manifest + migration framework (versioning Phases 2-3)."""
import json
import tempfile
import warnings
from pathlib import Path

import pytest
import polars as pl

from tcr_io import Dataset
from tcr_io.structure import (ARTIFACTS, Store, Manifest, Migrator, DATASET_VERSION,
                            PATIENT_META, PUBLICATION_IDS,
                            GENERATION_META, MANIFEST)
from tcr_io.structure import migrations
from tcr_io.structure.migrations.base import Migration
import tcr_io.structure.schema as S


def _make_dataset(manifest_version=None) -> Path:
    d = Path(tempfile.mkdtemp()) / "ds"
    Store(d).create_tree(ARTIFACTS)
    # Pre-v4 flat meta location; the v4 migration relocates it to meta/repertoire/TRB.parquet.
    pl.DataFrame(schema=S.REPERTOIRE_META).write_parquet(d / "meta/repertoire/repertoire.parquet")
    pl.DataFrame(schema=S.PATIENT_META).write_parquet(d / PATIENT_META.template)
    pl.DataFrame({"publication_id": []}, schema=S.PUBLICATION_META).write_ndjson(d / PUBLICATION_IDS.template)
    (d / GENERATION_META.template).write_text(json.dumps(
        {"dataset_name": "t", "created_on": "2026-01-01", "source": "x", "tcrio_version": "t",
         "reader": "r", "repertoire_mapper": "m", "patient_mapper": "m", "filters": {}}))
    if manifest_version is not None:
        Manifest(version=manifest_version).write(Store(d))
    return d


def test_first_migration_is_registered():
    assert 1 in migrations.REGISTRY


def test_v5_migration_backfills_clonotype_id():
    """v4->v5 adds a per-(repertoire, locus) clonotype_id (0..N-1) to each hive parquet.

    Migrates to 5 and stops. Every path and column here is a v4/v5-era literal — the same rule
    the migration modules themselves obey — because the live layout has moved twice since
    (v6 makes the shards locus-first, v7 rewrites the ingest record). Running the chain to head
    and then asserting a v5 shape tested a tree that no longer exists at head; end-to-end
    coverage is `tests/test_migration_chain.py`."""
    d = _make_dataset()
    (d / MANIFEST.template).write_text(json.dumps(          # v4-era literal, not the live class
        {"version": 4, "tcrio_version": "t", "present_loci": ["TRB"]}))
    f = d / "processed_repertoires/r1/locus=TRB/r1.parquet"   # v4-era path
    f.parent.mkdir(parents=True, exist_ok=True)
    # v4-shaped hive file: no clonotype_id column yet
    pl.DataFrame({
        "repertoire_id": ["r1"] * 3, "junction": ["A", "B", "C"], "v_call": ["TRBV2*01"] * 3,
        "junction_aa": ["CA", "CB", "CC"], "j_call": ["TRBJ1*01"] * 3,
        "duplicate_count": [3, 2, 1], "filter_pass": [True, True, False],
    }).write_parquet(f)

    Migrator(d).migrate(target=5)

    out = pl.read_parquet(f)
    assert out.columns[0] == "clonotype_id"          # leading column, per v5 REPERTOIRE order
    assert out["clonotype_id"].dtype == pl.UInt32
    assert out["clonotype_id"].to_list() == [0, 1, 2]   # contiguous within the file


def test_fresh_dataset_loads_clean():
    ds = _make_dataset(manifest_version=DATASET_VERSION)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Dataset(ds).version == DATASET_VERSION


def test_missing_manifest_is_v0_and_warns():
    ds = _make_dataset()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = Dataset(ds)
    assert d.version == 0
    assert any("migrated" in str(x.message) for x in w)


def test_dry_run_returns_plan_without_writing():
    ds = _make_dataset()
    d = Dataset(ds)
    plan = d.migrate(dry_run=True)
    assert [m.to_version for m in plan] == list(range(1, DATASET_VERSION + 1))
    assert not (Path(ds) / MANIFEST.template).exists()


def test_migrate_backfills_and_is_idempotent():
    ds = _make_dataset()
    d = Dataset(ds)
    d.migrate()
    assert d.version == DATASET_VERSION
    assert json.loads((Path(ds) / MANIFEST.template).read_text())["version"] == DATASET_VERSION
    assert d.migrate() == []          # no-op second time


def test_migrated_constructor_upgrades_without_warning():
    ds = _make_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = Dataset.migrated(ds)
    assert d.version == DATASET_VERSION


def test_future_version_raises():
    ds = _make_dataset(manifest_version=999)
    try:
        Dataset(ds)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_commit_per_step_resume():
    """A mid-chain failure leaves the manifest at the last committed version; re-run resumes."""
    ds = Dataset.migrated(_make_dataset())   # -> current DATASET_VERSION
    nxt = DATASET_VERSION + 1                    # a synthetic next migration
    try:
        migrations.REGISTRY[nxt] = Migration(nxt, "fails", lambda d: (_ for _ in ()).throw(RuntimeError("boom")))
        try:
            ds.migrate(target=nxt)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
        assert ds.version == DATASET_VERSION     # not committed past the failure

        applied = []
        migrations.REGISTRY[nxt] = Migration(nxt, "ok", lambda d: applied.append(1))
        ds.migrate(target=nxt)
        assert ds.version == nxt and applied == [1]
    finally:
        migrations.REGISTRY.pop(nxt, None)


def test_v4_migration_relocates_to_hive_layout():
    """v3->v4 moves flat processed parquets into {id}/locus=TRB/{id}.parquet and
    repertoire.parquet -> TRB.parquet, and records present_loci in the manifest.

    Migrates to 4 and stops, asserting v4-era literals — see the v5 test for why."""
    d = _make_dataset(manifest_version=3)
    # v3-shaped fixture: no clonotype_id column (that lands in v5). The v4 migration only relocates
    # the file; the v5 migration then backfills clonotype_id — so cast to the *current* REPERTOIRE
    # here would be wrong. Write the old shape as-is.
    pl.DataFrame(
        {"repertoire_id": ["r1"], "junction": ["TGT"], "v_call": ["TRBV2*01"],
         "junction_aa": ["CASSF"], "j_call": ["TRBJ2-1*01"], "duplicate_count": [3],
         "filter_pass": [True]}
    ).write_parquet(Path(d) / "processed_repertoires/r1.parquet")

    Migrator(d).migrate(target=4)       # stop at 4: v6 moves these shards again

    assert (Path(d) / "processed_repertoires/r1/locus=TRB/r1.parquet").exists()
    assert (Path(d) / "meta/repertoire/TRB.parquet").exists()
    assert not (Path(d) / "processed_repertoires/r1.parquet").exists()
    assert not (Path(d) / "meta/repertoire/repertoire.parquet").exists()
    # present_loci was a v4 manifest field; it left the manifest at v6 (the shard listing says
    # it), so this reads the raw JSON rather than the live class.
    assert json.loads((Path(d) / MANIFEST.template).read_text())["present_loci"] == ["TRB"]
    # hive read recovers locus from the path even though it's not in the file
    assert pl.scan_parquet(Path(d) / "processed_repertoires/r1", hive_partitioning=True) \
             .select("locus").unique().collect()["locus"].to_list() == ["TRB"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)


def test_migration_commits_only_the_version_field():
    """A migration sets `version` and preserves every other key — including keys THIS library
    version knows nothing about.

    Since v7 the manifest holds `version` alone, so there is no sibling field of ours left to
    clobber. The guard is now entirely about the future: a walk that rebuilt the manifest from
    the fields it knows would silently drop whatever a later version had written there, on
    exactly the trees a migration exists to carry forward."""
    d = _make_dataset(manifest_version=4)
    (d / MANIFEST.template).write_text(json.dumps({"version": 4, "from_the_future": ["keep me"]}))

    Migrator(d).migrate()

    raw = json.loads((d / MANIFEST.template).read_text())
    assert raw["version"] == DATASET_VERSION
    assert raw["from_the_future"] == ["keep me"]


def test_migrator_rejects_a_directory_that_is_not_there():
    with pytest.raises(FileNotFoundError):
        Migrator("/nonexistent/dataset")
