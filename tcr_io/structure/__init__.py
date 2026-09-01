"""Dataset structure: what a dataset holds, where it lives, and how it is read and written.

  schema.py   the column sets
  store.py    Artifact / Handle / Store — the only code that touches the filesystem
  layout.py   every well-known path, declared once as an Artifact
  version.py  the on-disk version and its manifest
  migrations/ upgrades between versions, each pinned to its own era's literals

Schemas are reached through the `schema` module (`schema.REPERTOIRE_META`), never re-exported
flat: half the artifacts share a name with the schema they carry, and the two are not
interchangeable.
"""
from . import schema, meta_edit
from .store import Artifact, Format, OnMissing, Handle, Store, resolve, required_dirs
from .layout import (
    ARTIFACTS, OPERATIONS_DIR,
    PROCESSED_DIR, LOCUS_DIR, REPERTOIRE_FILE,
    REPERTOIRE_META, REPERTOIRE_COUNTS, PATIENT_META, PUBLICATION_IDS, GENERATION_META,
    MANIFEST,
    REPERTOIRE_EXTRA, PATIENT_EXTRA, PUBLICATION_EXTRA, HLA,
    CLONE_TO_CELL_DIR, CLONE_TO_CELL,
    safe_repertoire_name,
)
from .version import DATASET_VERSION, Manifest
from .migrations import Migrator
