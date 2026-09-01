"""v6 -> v7: one ingest record, in `meta/generation.json`.

A v6 tree splits "how was this dataset made" across two files written five lines apart:

  - ``meta/generation.ndjson`` — a ONE-ROW NDJSON table: dataset_name, created_on, source,
    reader, repertoire_mapper, patient_mapper
  - ``meta/manifest.json`` — ``tcrio_version`` and ``filters`` (the per-locus filter
    provenance recorded at ingest), alongside ``version``

Only the manifest half had no reader. v7 merges both into a single JSON object at
``meta/generation.json`` and leaves the manifest holding ``version`` alone.

The format changes with the move. The record was always one row read back as
``.to_dicts()[0]`` — a record wearing a table's clothes — and ``filters`` is nested
(``{locus: {"preset", "filters": [names]}}``), which a polars schema can only express as a
struct column. A JSON object says both things directly.

Note the name: v6 renamed ``meta/generation.json`` (which held NDJSON content) to
``meta/generation.ndjson``. This step takes the name back, now that the content is JSON.

Idempotent: a tree whose ``meta/generation.json`` already parses as an object is left alone,
and the manifest keys are dropped whether or not they were there to drop.

Every path, key and column name below is a v7-era literal. None may come from the live layout
or schema: they describe a *v6* tree on the way to a *v7* one, both frozen. See base.py.
"""
import json
from pathlib import Path

from .base import migration

_V6_RECORD = "meta/generation.ndjson"
_V7_RECORD = "meta/generation.json"
_MANIFEST = "meta/manifest.json"

# The manifest keys that move. `version` stays.
_MOVED = ("tcrio_version", "filters")

# The v6 record's columns, in the order v7 writes them. `created_on` was a polars Date and
# serialises as an ISO string; the rest were already Utf8.
_V6_COLUMNS = ("dataset_name", "created_on", "source",
               "reader", "repertoire_mapper", "patient_mapper")


@migration(to_version=7,
           description="Merge the ingest record and manifest provenance into meta/generation.json")
def upgrade(db_dir: Path) -> None:
    record = _read_v6_record(db_dir)
    record.update(_take_from_manifest(db_dir))
    _write_v7_record(db_dir, record)


def _read_v6_record(db_dir: Path) -> dict:
    """The one row of `meta/generation.ndjson`, as a plain dict.

    Parsed with `json.loads` per line rather than polars: the file is one JSON object on one
    line, and reading it by hand keeps this step free of a frame library whose NDJSON reader
    has its own opinions about empty files and dtypes.

    A tree already carrying the v7 file (a resumed walk) hands that back instead — which is
    what makes the step idempotent.
    """
    v7 = db_dir / _V7_RECORD
    if v7.exists():
        loaded = json.loads(v7.read_text())
        if isinstance(loaded, dict):
            return loaded

    v6 = db_dir / _V6_RECORD
    if not v6.exists():
        return {}
    rows = [json.loads(line) for line in v6.read_text().splitlines() if line.strip()]
    row = rows[0] if rows else {}
    record = {k: row.get(k) for k in _V6_COLUMNS}
    # `created_on` may come back as a polars-written epoch-day int rather than a string.
    created = record.get("created_on")
    if isinstance(created, int):
        from datetime import date, timedelta
        record["created_on"] = (date(1970, 1, 1) + timedelta(days=created)).isoformat()
    return record


def _take_from_manifest(db_dir: Path) -> dict:
    """Pop `tcrio_version` and `filters` out of the manifest and return them.

    Read-modify-write on the raw dict, not through `Manifest`: a typed round-trip here would
    drop any key a LATER version adds, and this step must not assume it is the last one.
    """
    path = db_dir / _MANIFEST
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    moved = {k: data.pop(k) for k in _MOVED if k in data}
    path.write_text(json.dumps(data, indent=2) + "\n")
    return moved


def _write_v7_record(db_dir: Path, record: dict) -> None:
    """Write `meta/generation.json` and drop the NDJSON file it replaces.

    `filters` defaults to `{}`, not to the default preset: a dataset ingested before the
    provenance existed did not necessarily use it, and an empty map says "unrecorded" where a
    guess would say something false about how the data was cleaned.
    """
    record.setdefault("filters", {})
    (db_dir / _V7_RECORD).parent.mkdir(parents=True, exist_ok=True)
    (db_dir / _V7_RECORD).write_text(json.dumps(record, indent=2) + "\n")
    (db_dir / _V6_RECORD).unlink(missing_ok=True)
