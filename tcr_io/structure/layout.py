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
    JSON = "json"       # single pretty-printed object (e.g. the version manifest)


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
    processed_dir       = Artifact("processed_repertoires", Kind.DIR, description="one parquet per repertoire (atomic unit)")
    meta_dir            = Artifact("meta", Kind.DIR)
    repertoire_meta_dir = Artifact("meta/repertoire", Kind.DIR)
    patient_meta_dir    = Artifact("meta/patient", Kind.DIR)
    publication_dir     = Artifact("meta/publication", Kind.DIR)
    qc_dir              = Artifact("qc", Kind.DIR, description="QC metric tables + plots")
    tabulated_dir       = Artifact("tabulated", Kind.DIR, role=Role.GENERATED, description="hive-partitioned, rebuildable")

    # --- version manifest (absent on pre-versioning datasets -> optional) ---
    manifest = Artifact("meta/manifest.json", Kind.JSON, role=Role.OPTIONAL, description="dataset version")

    # --- core files (written at ingestion; carry schemas) ---
    generation_meta = Artifact("meta/generation.json", Kind.NDJSON, schema=schema.GENERATION_META, description="how/when/source this dataset was built")
    operations_meta = Artifact("meta/operations.json", Kind.NDJSON, schema=schema.OPERATIONS_META, description="one row per operation run")
    repertoire_meta = Artifact("meta/repertoire/repertoire.parquet", Kind.PARQUET, schema=schema.REPERTOIRE_META, description="one row per repertoire (ids, counts, source files)")
    patient_meta    = Artifact("meta/patient/patient.parquet", Kind.PARQUET, schema=schema.PATIENT_META, description="one row per patient")
    publication_ids = Artifact("meta/publication/publication_ids.json", Kind.NDJSON, schema=schema.PUBLICATION_META, description="DOIs / pubmed ids")

    # --- optional side files (joined in if present) ---
    repertoire_meta_extra = Artifact("meta/repertoire/repertoire_meta.parquet", Kind.PARQUET, role=Role.OPTIONAL, description="extra per-repertoire metadata, joined on repertoire_id")
    patient_meta_extra    = Artifact("meta/patient/patient_meta.parquet", Kind.PARQUET, role=Role.OPTIONAL, description="extra per-patient metadata, joined on patient_id")
    hla                   = Artifact("meta/patient/hla.parquet", Kind.PARQUET, role=Role.OPTIONAL, description="known HLA typing")
    inferred_hla          = Artifact("meta/patient/inferred_hla.parquet", Kind.PARQUET, role=Role.OPTIONAL, description="computationally inferred HLA")
    publication_meta      = Artifact("meta/publication/publication.parquet", Kind.PARQUET, role=Role.OPTIONAL, description="fetched publication metadata cache")


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


def _annotate(art: Optional[Artifact]) -> str:
    if art is None:
        return ""
    tags = []
    if art.role is not Role.REQUIRED:
        tags.append(art.role.value)
    if art.description:
        tags.append(art.description)
    return "  # " + " — ".join(tags) if tags else ""


def render_tree(root_name: str = "dataset", examples: bool = True) -> str:
    """Render the layout registry as an ASCII directory tree.

    The tree is *derived* from `LAYOUT`, so it can never drift from the real structure
    (that's the point). `examples=True` also shows the parameterised per-repertoire file,
    which lives in a path builder rather than a static Artifact.
    """
    # nested trie: part -> {"art": Artifact|None, "children": {...}}
    tree: dict = {}
    entries = [(a.path, a) for a in LAYOUT.values()]
    if examples:
        entries.append((repertoire_relpath("sample_001"), None))  # illustrative leaf

    for path, art in entries:
        node = tree
        parts = path.split("/")
        for i, part in enumerate(parts):
            entry = node.setdefault(part, {"art": None, "children": {}})
            if i == len(parts) - 1 and art is not None:
                entry["art"] = art
            node = entry["children"]

    def is_dir(entry) -> bool:
        return bool(entry["children"]) or (entry["art"] is not None and entry["art"].kind is Kind.DIR)

    lines = [f"{root_name}/"]

    def walk(node: dict, prefix: str) -> None:
        # directories first, then alphabetically
        items = sorted(node.items(), key=lambda kv: (not is_dir(kv[1]), kv[0]))
        for i, (name, entry) in enumerate(items):
            last = i == len(items) - 1
            connector = "└── " if last else "├── "
            label = name + ("/" if is_dir(entry) else "")
            lines.append(f"{prefix}{connector}{label}{_annotate(entry['art'])}")
            if entry["children"]:
                walk(entry["children"], prefix + ("    " if last else "│   "))

    walk(tree, "")
    return "\n".join(lines)


if __name__ == "__main__":   # python -m tcr_io.structure.layout
    print(render_tree())
