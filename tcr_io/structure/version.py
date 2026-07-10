"""Dataset version manifest.

A single incremental `DATASET_VERSION` is recorded on disk in `meta/manifest.json`,
written at ingestion and checked on load. The manifest is modelled as a frozen dataclass
whose defaults *are* the pre-versioning (v0) state, so an absent file just constructs the
default -- "no manifest => v0" needs no special-casing.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
import json
from pathlib import Path

from .._internal import __version__ as _tcrio_version

DATASET_VERSION = 3   # v3: hive outputs folded into unstructured op-written dirs; temp/ dropped
                      # v2: all operation outputs live under operations/ (per-op operation.json);
                      # v1 = the pre-restructure layout (scattered op outputs + meta/operations.json)


@dataclass(frozen=True)
class Manifest:
    version: int = 0          # 0 == pre-versioning dataset (no manifest on disk)
    tcrio_version: str = ""   # library version that last wrote it

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        if not path.exists():
            return cls()                       # <- "no manifest => v0"
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})   # forward-compatible

    @classmethod
    def current(cls) -> "Manifest":
        return cls(version=DATASET_VERSION, tcrio_version=_tcrio_version)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
