"""Dataset version manifest.

A single incremental `DATASET_VERSION` is recorded on disk in `meta/manifest.json`,
written at ingestion and checked on load. The manifest is modelled as a frozen dataclass
whose defaults *are* the pre-versioning (v0) state, so an absent file just constructs the
default -- "no manifest => v0" needs no special-casing.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields

from .layout import MANIFEST
from .store import Store

DATASET_VERSION = 7   # v7: the ingest record moves to `meta/generation.json` (a JSON object, not a
                      #     one-row NDJSON table) and absorbs `tcrio_version` and `filters` from the
                      #     manifest — one record answering "how was this dataset made". The manifest
                      #     is left holding `version` alone.
                      # v6: locus-first shards (processed_repertoires/locus={L}/{id}.parquet); repertoire
                      #     meta split by grain (meta/repertoire/repertoire.parquet +
                      #     locus={L}/counts.parquet); side tables renamed to extra.parquet; the two
                      #     NDJSON files renamed off `.json`; `filter_pass` -> `filter_reason`;
                      #     `present_loci` dropped from the manifest (it is the shard listing);
                      #     operations/ cleared, since every result is regenerable.
                      # v5: per-(repertoire, locus) `clonotype_id` in REPERTOIRE (row number within each
                      #     locus partition); single-cell datasets also write meta/clone_to_cell/{id}.parquet
                      #     (repertoire_id, locus, clonotype_id, cell_id). Migration backfills clonotype_id
                      #     per hive file.
                      # v4: per-repertoire hive layout — processed_repertoires/{id}/locus={LOCUS}/{id}.parquet
                      #     (locus is the hive partition, not stored in the file); written in one
                      #     pl.PartitionBy streaming pass. `present_loci` recorded in the manifest.
                      #     repertoire meta stored one parquet per locus (meta/repertoire/{LOCUS}.parquet).
                      # v3: hive outputs folded into unstructured op-written dirs; temp/ dropped
                      # v2: all operation outputs live under operations/ (per-op operation.json);
                      # v1 = the pre-restructure layout (scattered op outputs + meta/operations.json)

IMGT_VERSION = "202614-2 (31 March 2026)"  # data releseas: https://www.imgt.org/IMGT_vquest/data_releases


@dataclass(frozen=True)
class Manifest:
    """The version, and nothing else.

    `tcrio_version` and `filters` used to live here. Both describe how a dataset was *made*,
    which is the ingest record's question (`meta/generation.json`) — and neither had a reader
    while it sat here. The manifest is for what the tree cannot say about itself, and the
    version is the whole of that.
    """
    version: int = 0                    # 0 == pre-versioning dataset (no manifest on disk)

    @classmethod
    def read(cls, store: Store) -> "Manifest":
        """`MANIFEST` is `on_missing=NONE`, so an absent file reads as `None` and constructs
        the defaults — "no manifest => v0" needs no branch here."""
        known = {f.name for f in fields(cls)}
        data = store(MANIFEST).read() or {}
        return cls(**{k: v for k, v in data.items() if k in known})   # forward-compatible

    @classmethod
    def current(cls) -> "Manifest":
        return cls(version=DATASET_VERSION)

    def write(self, store: Store) -> None:
        """Whole-value write, for the one writer that holds the whole value: ingestion.

        `Migrator` still commits through `store(MANIFEST).update(...)`. Not because ownership
        is split any more — it is not — but because a walk must preserve keys a LATER version
        adds, which this version cannot know to carry."""
        store(MANIFEST).write(asdict(self))
