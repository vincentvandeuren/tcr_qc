"""Version manifest + migration framework (versioning Phases 2-3)."""
import json
import tempfile
import warnings
from pathlib import Path

import polars as pl

from tcr_io import TcrDataset
from tcr_io.layout import Layout, REQUIRED_DIRS, GENERATED_DIRS
from tcr_io.version import Manifest, DATASET_VERSION
from tcr_io import migrations
from tcr_io.migrations.base import Migration
import tcr_io.schema as S


def _make_dataset(manifest_version=None) -> Path:
    d = Path(tempfile.mkdtemp()) / "ds"
    for sub in REQUIRED_DIRS + GENERATED_DIRS:
        (d / sub).mkdir(parents=True, exist_ok=True)
    pl.DataFrame(schema=S.REPERTOIRE_META).write_parquet(d / Layout.repertoire_meta.path)
    pl.DataFrame(schema=S.PATIENT_META).write_parquet(d / Layout.patient_meta.path)
    pl.DataFrame({"publication_id": []}, schema=S.PUBLICATION_META).write_ndjson(d / Layout.publication_ids.path)
    pl.DataFrame(
        {"dataset_name": ["t"], "created_on": [None], "source": ["x"],
         "reader": ["r"], "repertoire_mapper": ["m"], "patient_mapper": ["m"]}
    ).cast(S.GENERATION_META).write_ndjson(d / Layout.generation_meta.path)
    pl.DataFrame(schema=S.OPERATIONS_META).write_ndjson(d / Layout.operations_meta.path)
    if manifest_version is not None:
        Manifest(version=manifest_version, tcrio_version="t").write(d / Layout.manifest.path)
    return d


def test_first_migration_is_registered():
    assert 1 in migrations.REGISTRY


def test_fresh_dataset_loads_clean():
    ds = _make_dataset(manifest_version=DATASET_VERSION)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert TcrDataset(ds).version == DATASET_VERSION


def test_missing_manifest_is_v0_and_warns():
    ds = _make_dataset()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = TcrDataset(ds)
    assert d.version == 0
    assert any("migrated" in str(x.message) for x in w)


def test_dry_run_returns_plan_without_writing():
    ds = _make_dataset()
    d = TcrDataset(ds)
    plan = d.migrate(dry_run=True)
    assert [m.to_version for m in plan] == [1]
    assert not (Path(ds) / Layout.manifest.path).exists()


def test_migrate_backfills_and_is_idempotent():
    ds = _make_dataset()
    d = TcrDataset(ds)
    d.migrate()
    assert d.version == 1
    assert json.loads((Path(ds) / Layout.manifest.path).read_text())["version"] == 1
    assert d.migrate() == []          # no-op second time


def test_migrated_constructor_upgrades_without_warning():
    ds = _make_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = TcrDataset.migrated(ds)
    assert d.version == DATASET_VERSION


def test_future_version_raises():
    ds = _make_dataset(manifest_version=999)
    try:
        TcrDataset(ds)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_commit_per_step_resume():
    """A mid-chain failure leaves the manifest at the last committed version; re-run resumes."""
    ds = TcrDataset.migrated(_make_dataset())   # -> v1
    try:
        migrations.REGISTRY[2] = Migration(2, "fails", lambda d: (_ for _ in ()).throw(RuntimeError("boom")))
        try:
            ds.migrate(target=2)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
        assert ds.version == 1                  # not committed past the failure

        applied = []
        migrations.REGISTRY[2] = Migration(2, "ok", lambda d: applied.append(1))
        ds.migrate(target=2)
        assert ds.version == 2 and applied == [1]
    finally:
        migrations.REGISTRY.pop(2, None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
