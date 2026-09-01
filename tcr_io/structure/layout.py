"""Every well-known path in a dataset, declared once.

An artifact is a *template* plus everything needed to read and write it: its format
(extension + codec), its schema, and what absence means. Nothing else in the library
constructs a dataset path — call sites resolve through a `Store`:

    store = Store(db_dir)
    store(PATIENT_META).read()
    store(REPERTOIRE_FILE, repertoire_id="s1", locus="TRB").path()
    store(REPERTOIRE_FILE, locus="TRB").glob()        # repertoire_id unbound -> every match

The tree is locus-first: `locus={L}/` is one level under `processed_repertoires/`, so the
hot path — every file of one locus — is a single listing, `rm -r
processed_repertoires/locus=TRB/` means something, and the set of loci a dataset holds is a
listing rather than a value copied into the manifest.
"""
from __future__ import annotations

import polars as pl

from . import schema as s
from .store import Artifact, Format as F, OnMissing as M

# --- repertoires ------------------------------------------------------------------------
# `locus` lives only in the path (`include_key=False`), recovered on read via hive
# partitioning. The leaves are written by one `pl.PartitionBy` sink at ingest, so they are
# not individually writable.

PROCESSED_DIR   = Artifact("processed_repertoires", F.PARQUET_DIR, writable=False)
LOCUS_DIR       = Artifact("processed_repertoires/locus={locus}", F.PARQUET_DIR, writable=False)
REPERTOIRE_FILE = Artifact("processed_repertoires/locus={locus}/{repertoire_id}.parquet",
                           F.PARQUET, s.REPERTOIRE, writable=False)

# --- metadata written at ingestion ------------------------------------------------------
# Repertoire metadata is split by grain: what is true of a repertoire whatever the locus
# (`REPERTOIRE_META`) and what is only true within one (`REPERTOIRE_COUNTS`). One table
# carrying both duplicated the locus-invariant half into every locus file, and every
# consumer of that half paid to read the other and then dedupe it away.

REPERTOIRE_META   = Artifact("meta/repertoire/repertoire.parquet", F.PARQUET, s.REPERTOIRE_META)
REPERTOIRE_COUNTS = Artifact("meta/repertoire/locus={locus}/counts.parquet",
                             F.PARQUET, s.REPERTOIRE_COUNTS)
PATIENT_META      = Artifact("meta/patient/patient.parquet",    F.PARQUET, s.PATIENT_META)
PUBLICATION_IDS   = Artifact("meta/publication/publication_ids.ndjson", F.NDJSON, s.PUBLICATION_META)
GENERATION_META   = Artifact("meta/generation.json",           F.JSON)
MANIFEST          = Artifact("meta/manifest.json",              F.JSON,   on_missing=M.NONE)

# --- optional side tables, joined in when present ---------------------------------------
# `requires` is the join key: these carry user columns under no fixed schema, so the only
# thing worth asserting is that the key is there and is the right type. Named `extra.parquet`
# under the grain they extend, so `meta/repertoire/` reads as one directory about
# repertoires rather than as two files whose names differ by a suffix.

REPERTOIRE_EXTRA  = Artifact("meta/repertoire/extra.parquet", F.PARQUET,
                             on_missing=M.NONE, requires=pl.Schema({"repertoire_id": pl.Utf8}))
PATIENT_EXTRA     = Artifact("meta/patient/extra.parquet", F.PARQUET,
                             on_missing=M.NONE, requires=pl.Schema({"patient_id": pl.Utf8}))
PUBLICATION_EXTRA = Artifact("meta/publication/publication.parquet", F.PARQUET,
                             on_missing=M.NONE, requires=pl.Schema({"publication_id": pl.Utf8}))
HLA               = Artifact("meta/patient/hla.parquet", F.PARQUET, s.HLA_META, on_missing=M.NONE)

# --- single-cell clonotype<->cell map (single-cell datasets only) ------------------------
# EMPTY, not NONE: a bulk repertoire has no map, and an empty typed frame joins correctly
# where a `None` would need a branch at every call site.

CLONE_TO_CELL_DIR = Artifact("meta/clone_to_cell", F.PARQUET_DIR, writable=False,
                             on_missing=M.NONE)
CLONE_TO_CELL     = Artifact("meta/clone_to_cell/{repertoire_id}.parquet",
                             F.PARQUET, s.CLONE_TO_CELL, on_missing=M.EMPTY)

ARTIFACTS = (
    PROCESSED_DIR, LOCUS_DIR, REPERTOIRE_FILE,
    REPERTOIRE_META, REPERTOIRE_COUNTS, PATIENT_META, PUBLICATION_IDS, GENERATION_META,
    MANIFEST,
    REPERTOIRE_EXTRA, PATIENT_EXTRA, PUBLICATION_EXTRA, HLA,
    CLONE_TO_CELL_DIR, CLONE_TO_CELL,
)

assert len({a.template for a in ARTIFACTS}) == len(ARTIFACTS), "duplicate artifact template"

# Every artifact's name matches its format. This used to carry an exemption for two NDJSON
# files named `.json`; the v6 rename removed them, and the exemption died with them.
assert not [a.template for a in ARTIFACTS
            if not a.format.is_dir and not a.template.endswith(a.format.ext)]

# Operation outputs live under one generated root, rooted and named by the runner rather
# than declared here: their paths are `(operation, locus, output name)`, not fixed strings.
OPERATIONS_DIR = "operations"


def safe_repertoire_name(repertoire_id: str) -> str:
    """Filesystem-safe repertoire filename stem."""
    return repertoire_id.replace("/", "_")
