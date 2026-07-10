"""Kind-keyed write dispatch for Artifacts.

Symmetric with the read side: a value is written according to its Artifact's `Kind`
(not re-derived per call site), with an optional schema-cast on the way out so the
on-disk file is guaranteed to match the schema a later reader enforces.

This lifts the value-`match` that used to live in `dataset._write_operation_results`
into one reusable writer, shared by the static ingestion path and operation outputs.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Union
import polars as pl

from .layout import Artifact, Kind


def _write_parquet(result, path: Path) -> None:
    if isinstance(result, pl.LazyFrame):
        result.sink_parquet(path)
    else:
        result.write_parquet(path)


def _write_ndjson(result, path: Path) -> None:
    # `result` is a list of json-serialisable records (dicts).
    with open(path, "w") as f:
        for record in result:
            f.write(json.dumps(record, default=str) + "\n")


def _write_json(result, path: Path) -> None:
    # `result` is a single json-serialisable object (dict / list / scalar).
    path.write_text(json.dumps(result, indent=2, default=str))


# Note: there is no writer for `Kind.UNSTRUCTURED` — those directory outputs are written by
# the operation itself into the managed dir from `TcrDataset._operation_output_dir`, not by
# the framework (see docs/unstructured_output_plan.md).
_WRITERS = {
    Kind.PARQUET: _write_parquet,
    Kind.NDJSON: _write_ndjson,
    Kind.JSON: _write_json,
}


def write_artifact(art: Artifact, result, base_dir: Union[str, Path]) -> Path:
    """Write `result` to `base_dir / art.path`, dispatching on `art.kind`.

    When the Artifact carries a schema and the value is a frame, the frame is
    column-selected and cast to that schema first (validate-on-write). Returns the
    absolute path written.
    """
    dst = Path(base_dir) / art.path
    dst.parent.mkdir(parents=True, exist_ok=True)

    if art.schema is not None and isinstance(result, (pl.DataFrame, pl.LazyFrame)):
        result = result.select(art.schema.keys()).cast(art.schema)

    writer = _WRITERS.get(art.kind)
    if writer is None:
        raise ValueError(f"No writer registered for {art.kind} (artifact {art.path!r})")
    writer(result, dst)
    return dst
