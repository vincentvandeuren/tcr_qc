"""Dataset version manifest.

A single incremental `DATASET_VERSION` is recorded on disk in `meta/manifest.json`,
written at ingestion and checked on load. The manifest is modelled as a frozen dataclass
whose defaults *are* the pre-versioning (v0) state, so an absent file just constructs the
default -- "no manifest => v0" needs no special-casing.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields
import json
from pathlib import Path
from typing import List

from .._internal import __version__ as _tcrio_version

DATASET_VERSION = 4   # v4: per-repertoire hive layout — processed_repertoires/{id}/locus={LOCUS}/{id}.parquet
                      #     (locus is the hive partition, not stored in the file); written in one
                      #     pl.PartitionBy streaming pass. `present_loci` recorded in the manifest.
                      #     repertoire meta stored one parquet per locus (meta/repertoire/{LOCUS}.parquet).
                      # v3: hive outputs folded into unstructured op-written dirs; temp/ dropped
                      # v2: all operation outputs live under operations/ (per-op operation.json);
                      # v1 = the pre-restructure layout (scattered op outputs + meta/operations.json)


@dataclass(frozen=True)
class Manifest:
    version: int = 0                    # 0 == pre-versioning dataset (no manifest on disk)
    tcrio_version: str = ""             # library version that last wrote it
    present_loci: List[str] = field(default_factory=list)   # loci in this dataset, fixed at ingest

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        if not path.exists():
            return cls()                       # <- "no manifest => v0"
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})   # forward-compatible

    @classmethod
    def current(cls, present_loci: List[str] | None = None) -> "Manifest":
        return cls(version=DATASET_VERSION, tcrio_version=_tcrio_version,
                   present_loci=sorted(present_loci or []))

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
