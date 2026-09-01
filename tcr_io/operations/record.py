"""The operation ledger: what an operation wrote, and what it wrote it with.

One record per **(operation, locus)**, at `operations/<op>/<locus>/operation.json`.
The record's *location* is half
its value: it sits inside the directory it describes, so clearing an output set and
clearing its description are one act.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from ..structure import Artifact, Format, Handle, OnMissing

# Declared relative to an operation's output Store, NOT to the dataset root: the record binds
# through the very object its outputs bind through, so it cannot land anywhere else than beside
# them. A root-relative `operations/{operation}/{locus}/operation.json` would re-derive the
# same path from three parameters the runner has already resolved once.
OPERATION_RECORD = Artifact("operation.json", Format.JSON, on_missing=OnMissing.NONE)


def serialize_params(params: dict) -> dict:
    """The ONE serialization path for operation params — used when writing a record and again
    when comparing a prior record's params to a live op's.

    Both sides go through here, or the comparison is between a dict and its own lossy image.
    There is deliberately no `default=str` fallback: it turned a `FisherTest` into its repr,
    so `rebuild` reconstructed params of the wrong type and nothing raised. An unserialisable
    param is an error *here*, where the operation that declared it can still be fixed.
    """
    return json.loads(json.dumps(params, default=_encode))


def _encode(v: Any):
    if is_dataclass(v) and not isinstance(v, type):
        return asdict(v)          # round-trips: ops coerce dicts back in __post_init__
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    raise TypeError(
        f"operation param of type {type(v).__name__!r} is not serialisable; make it a "
        f"dataclass, a Path, or a plain JSON value so `rebuild` can reconstruct it"
    )


@dataclass
class OutputRecord:
    """One file (or directory) an operation wrote.

    Self-describing, so reading it back never needs the operation class to be importable —
    which fails exactly when you most want the data. No `locus` field: the enclosing record
    is already scoped to one locus.
    """
    name: str            # the Artifact's class-attribute name
    template: str        # resolved, relative to the record's own directory
    format: str          # Format member name

    @classmethod
    def from_dict(cls, d: dict) -> "OutputRecord":
        return cls(name=d["name"], template=d["template"], format=d["format"])

    def artifact(self) -> Artifact:
        """An Artifact good enough to read this back without the declaring class. The schema
        is not recorded, so this one has none — a read through it is uncast."""
        return Artifact(self.template, Format[self.format], on_missing=OnMissing.NONE)


@dataclass
class OperationRecord:
    """ONE OPERATION x ONE LOCUS.

    A *historical* record: it stores what was written rather than deriving paths from the
    class. The drift risk that buys is contained by one rule — **the record is never
    consulted when writing**.

    Per-locus rather than per-operation, for three reasons, none cosmetic:

      partial failure   TRB succeeds, TRA raises. A shared record needs one `status` for
                        both, which is why the old code merged with the prior record to
                        salvage TRB. Split records have nothing to merge.
      clear() coheres   the runner clears `operations/<op>/<locus>/`. A record one level up
                        would survive, still listing files that no longer exist.
      honest staleness  bump the version, re-run TRB alone. A shared record would claim the
                        new version for TRA's files too and hand back stale data silently.
                        Each locus carrying the version and params that produced ITS
                        files makes that raise instead.

    Cost: `ds.operations` needs two globs.
    """
    operation_name: str
    version: str
    description: str
    ran_at: datetime
    duration_s: float
    status: str                     # "success" | "failure"
    error: Optional[str]
    params: dict
    locus: str                      # informational; the directory is the authority
    outputs: List[OutputRecord]

    @classmethod
    def read(cls, h: Handle) -> Optional["OperationRecord"]:
        """None when absent — `OPERATION_RECORD` is `on_missing=NONE`, so there is no branch."""
        d = h.read()
        return cls.from_dict(d) if d else None

    def write(self, h: Handle) -> None:
        """Whole-value write, never `update()`. This record is single-writer (the runner,
        which always holds the complete value), so merging could only lose information."""
        h.write(json.loads(json.dumps(asdict(self), default=str)))

    @classmethod
    def from_dict(cls, d: dict) -> "OperationRecord":
        return cls(
            operation_name=d["operation_name"],
            version=d["version"],
            description=d.get("description", ""),
            ran_at=d.get("ran_at"),
            duration_s=d.get("duration_s"),
            status=d["status"],
            error=d.get("error"),
            params=d.get("params") or {},
            locus=d.get("locus"),
            outputs=[OutputRecord.from_dict(o) for o in d.get("outputs", [])],
        )
