"""
Single source of truth for the dataset folder structure.

Every well-known path in a dataset is declared once here as an `Artifact` on the
`Layout` class. Read/write sites resolve paths through `Layout.<key>.path` (or
`LAYOUT["<key>"]`) instead of hardcoding literals, so the structure can no longer
drift between `dataset.py`, `ingestion.py`, and the operations.

Per-repertoire files can't be enumerated (one per repertoire), so they get a path
*builder* (`repertoire_relpath`) rather than a static `Artifact`. The builder is the
single place the BCR locus split adds a `{LOCUS}/` segment, and where single-cell adds
`clone_to_cell` — see docs/dataset_versioning_plan.md and the BCR / single-cell plans.

Operation outputs remain free-form (open-ended), but should reference the well-known
roots below (`Layout.repertoire_meta_dir.path`, `Layout.qc_dir.path`, ...).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import polars as pl

from . import schema


class Kind(Enum):
    DIR = "dir"
    PARQUET = "parquet"
    NDJSON = "ndjson"


class Role(Enum):
    REQUIRED = "required"      # validated on load, created at ingestion
    OPTIONAL = "optional"      # joined in if present
    GENERATED = "generated"    # created at ingestion, rebuildable/deletable, not validated


@dataclass(frozen=True)
class Artifact:
    path: str
    kind: Kind
    role: Role = Role.REQUIRED
    schema: Optional[pl.Schema] = None
    description: str = ""
    key: str = field(default="", compare=False)   # filled in by __set_name__

    def __set_name__(self, owner, name):
        object.__setattr__(self, "key", name)      # frozen dataclass -> bypass __setattr__
        owner._registry[name] = self


class Layout:
    """Declarative dataset structure. Access as `Layout.repertoire_meta.path`."""
    _registry: dict[str, Artifact] = {}

    # --- directories (the skeleton) ---
    processed_dir       = Artifact("processed_repertoires", Kind.DIR)
    meta_dir            = Artifact("meta", Kind.DIR)
    repertoire_meta_dir = Artifact("meta/repertoire", Kind.DIR)
    patient_meta_dir    = Artifact("meta/patient", Kind.DIR)
    publication_dir     = Artifact("meta/publication", Kind.DIR)
    qc_dir              = Artifact("qc", Kind.DIR)
    tabulated_dir       = Artifact("tabulated", Kind.DIR, role=Role.GENERATED)

    # --- core files (written at ingestion; carry schemas) ---
    generation_meta = Artifact("meta/generation.json", Kind.NDJSON, schema=schema.GENERATION_META)
    operations_meta = Artifact("meta/operations.json", Kind.NDJSON, schema=schema.OPERATIONS_META)
    repertoire_meta = Artifact("meta/repertoire/repertoire.parquet", Kind.PARQUET, schema=schema.REPERTOIRE_META)
    patient_meta    = Artifact("meta/patient/patient.parquet", Kind.PARQUET, schema=schema.PATIENT_META)
    publication_ids = Artifact("meta/publication/publication_ids.json", Kind.NDJSON, schema=schema.PUBLICATION_META)

    # --- optional side files (joined in if present) ---
    repertoire_meta_extra = Artifact("meta/repertoire/repertoire_meta.parquet", Kind.PARQUET, role=Role.OPTIONAL)
    patient_meta_extra    = Artifact("meta/patient/patient_meta.parquet", Kind.PARQUET, role=Role.OPTIONAL)
    hla                   = Artifact("meta/patient/hla.parquet", Kind.PARQUET, role=Role.OPTIONAL)
    inferred_hla          = Artifact("meta/patient/inferred_hla.parquet", Kind.PARQUET, role=Role.OPTIONAL)
    publication_meta      = Artifact("meta/publication/publication.parquet", Kind.PARQUET, role=Role.OPTIONAL)


LAYOUT: dict[str, Artifact] = Layout._registry

REQUIRED_DIRS  = [a.path for a in LAYOUT.values() if a.kind is Kind.DIR and a.role is Role.REQUIRED]
GENERATED_DIRS = [a.path for a in LAYOUT.values() if a.kind is Kind.DIR and a.role is Role.GENERATED]


def safe_repertoire_name(repertoire_id: str) -> str:
    """Filesystem-safe repertoire filename stem."""
    return repertoire_id.replace("/", "_")


def repertoire_relpath(repertoire_id: str) -> str:
    """Relative path of a repertoire's processed parquet.

    The BCR locus split adds a `locus` parameter + a `{LOCUS}/` segment here later;
    keeping construction in one place means that becomes a one-line change.
    """
    return f"{Layout.processed_dir.path}/{safe_repertoire_name(repertoire_id)}.parquet"
