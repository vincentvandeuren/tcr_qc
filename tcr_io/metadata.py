"""`MetadataWriter` — every supported edit to an existing dataset's metadata.

Split off the read class because these are the only writes a *user* makes to a dataset that
already exists, and they share one validation path that has nothing to do with reading: keys
must be unique, must name entities the dataset actually has, and must not collide with a
column the library owns.

Layering: the frame algebra lives one level down in `structure.meta_edit`, side-effect-free and
unit-testable without a dataset on disk. This module is the half that does I/O and decides
warn-versus-raise. It reads through a `Dataset` rather than a bare `Store` because validation
needs the canonical tables the new rows are checked against.

Reached as `ds.metadata`:

    ds.metadata.set_patient(df)                    # merge: add columns/rows, touch nothing
    ds.metadata.set_repertoire(df, mode="replace") # overwrite the table
    ds.metadata.set_hla(df)                        # always a replace; fixed schema
    ds.metadata.fetch_publications()               # derives its rows from the web
"""
from __future__ import annotations

import warnings
from dataclasses import asdict
from typing import TYPE_CHECKING, Iterable, List, Optional

import polars as pl

from .publications import fetch_publication
from .structure import layout, Artifact
from .structure import meta_edit, schema as _schema

if TYPE_CHECKING:                       # type-only: no runtime import, no cycle
    from .dataset import Dataset


def _fmt_ids(ids: Iterable[str], cap: int = 10) -> str:
    """Render a (possibly long) id list for a warning, capped with an ellipsis."""
    ids = list(ids)
    shown = [str(i) for i in ids[:cap]]
    if len(ids) > cap:
        shown.append(f"... (+{len(ids) - cap} more)")
    return "[" + ", ".join(shown) + "]"


class MetadataWriter:
    """Writes the optional side tables that `full_*_meta` joins in.

    Ideally used during or right after ingestion, but supported at any time. See
    docs/superpowers/specs/2026-07-15-metadata-write-api-design.md.
    """

    def __init__(self, ds: "Dataset"):
        self._ds = ds

    # --- free-form side tables ----------------------------------------------------------
    # Three tables, three keys, one write path. Spelled out rather than dispatched: the whole
    # difference between them is three arguments, and a table mapping name -> (artifact, key,
    # base) would hide that behind a lookup nothing else needs.

    def set_patient(self, df: pl.DataFrame, *, mode: str = "merge") -> None:
        """Extra per-patient metadata (keyed on ``patient_id``), joined into
        ``full_patient_meta``. ``mode='merge'`` (default) adds new columns and rows without
        touching existing columns; ``mode='replace'`` overwrites the table."""
        self._set_free_form(df, key="patient_id", art=layout.PATIENT_EXTRA, mode=mode,
                            base=self._ds.store(layout.PATIENT_META).read(), label="set_patient")

    def set_repertoire(self, df: pl.DataFrame, *, mode: str = "merge") -> None:
        """Extra per-repertoire metadata (keyed on ``repertoire_id``), joined into
        ``full_repertoire_meta`` across every locus. The table is locus-agnostic — one row per
        repertoire — so there is no ``locus`` argument."""
        self._set_free_form(df, key="repertoire_id", art=layout.REPERTOIRE_EXTRA, mode=mode,
                            base=self._ds.repertoire_meta, label="set_repertoire")

    def set_publication(self, df: pl.DataFrame, *, mode: str = "merge") -> None:
        """Extra per-publication metadata (keyed on ``publication_id``), joined into
        ``full_publication_meta``. The reader has always joined this table; until now nothing
        could write it."""
        self._set_free_form(df, key="publication_id", art=layout.PUBLICATION_EXTRA, mode=mode,
                            base=self._ds.store(layout.PUBLICATION_IDS).read(), label="set_publication")

    def _set_free_form(self, df: pl.DataFrame, *, key: str, art: Artifact, base: pl.DataFrame,
                       mode: str, label: str) -> None:
        """Validate, then write. Duplicate keys raise; unknown or missing keys warn; columns
        already present in the base or existing extra table warn and are dropped.

        The merge is *additive* by design: it adds columns and rows but never rewrites a column
        that exists, which is why collisions are stripped before the join rather than resolved
        inside it. Overwriting is `mode='replace'`, spelled out by the caller."""
        if mode not in ("merge", "replace"):
            raise ValueError(f"{label}: mode must be 'merge' or 'replace', got {mode!r}.")
        if key not in df.columns:
            raise ValueError(f"{label}: input is missing the '{key}' key column.")

        self._validate_keys(df, key=key, base_keys=base.get_column(key).unique().to_list(),
                            label=label)

        existing = self._ds.store(art).read() if mode == "merge" else None

        reserved = list(base.columns) + (existing.columns if existing is not None else [])
        if collisions := meta_edit.reserved_column_collisions(df.columns, reserved, key):
            warnings.warn(
                f"{label}: column(s) {collisions} already exist in the metadata and were skipped; "
                f"use mode='replace' to overwrite them.",
                stacklevel=3,
            )
            df = df.drop(collisions)

        self._ds.store(art).write(df if existing is None
                                  else meta_edit.merge_frames(existing, df, key))

    # --- fetched, rather than supplied ---------------------------------------------------

    def fetch_publications(self) -> pl.DataFrame:
        """Look each ``publication_id`` up on the web and store what comes back — title,
        abstract, authors, year, journal, and the DOI/PMID it resolved to — joined into
        ``full_publication_meta``.

        This is the one writer that *derives* its rows instead of accepting them, so it is an
        explicit call and never something a read triggers: it is one HTTP request per
        identifier (several for a BioProject), against rate-limited public APIs.

        Identifiers are recognized by shape — DOI, PMC id, BioProject accession, immuneACCESS
        URL. One that cannot be recognized or cannot be reached gets a row with ``error`` set
        rather than no row, so a blank publication can say why it is blank.

        Re-running refreshes: the columns in `schema.PUBLICATION_FETCHED` belong to this
        method and are replaced wholesale, while columns added by `set_publication` are kept.
        (`set_publication` refuses the reverse — it warns and skips a column that already
        exists — so the rule is the same from both sides: the fetched value wins for fetched
        columns.)
        """
        ids = self._ds.store(layout.PUBLICATION_IDS).read().get_column("publication_id").to_list()
        fetched = pl.DataFrame(
            [{**asdict(fetch_publication(pub_id)), "publication_id": pub_id} for pub_id in ids],
            schema=_schema.PUBLICATION_FETCHED,
        )

        # Drop this method's own columns from what is already there, then merge the rest back:
        # `merge_frames` is additive and would otherwise refuse to refresh a stale title.
        existing = self._ds.store(layout.PUBLICATION_EXTRA).read()
        if existing is not None:
            kept = existing.drop([c for c in fetched.columns
                                  if c != "publication_id" and c in existing.columns])
            fetched = meta_edit.merge_frames(kept, fetched, "publication_id")

        self._ds.store(layout.PUBLICATION_EXTRA).write(fetched)
        return fetched

    # --- HLA: a schema'd table, so a different shape of write ---------------------------

    def set_hla(self, df: pl.DataFrame) -> None:
        """Ground-truth HLA typing (keyed on ``patient_id``), joined into ``full_patient_meta``.

        Always a full **replace**: `schema.HLA_META`'s fixed eight columns leave no room for a
        column-adding merge. Input must already be in canonical notation — bare 4-digit allele
        strings per locus. Absent loci are filled with null lists (partial typing is allowed),
        columns outside the schema are warned and dropped, and the write casts to `HLA_META`,
        so a wrong dtype raises."""
        key, label = "patient_id", "set_hla"
        if key not in df.columns:
            raise ValueError(f"{label}: input is missing the '{key}' key column.")

        base = self._ds.store(layout.PATIENT_META).read()
        self._validate_keys(df, key=key, base_keys=base.get_column(key).to_list(), label=label)

        if extra := [c for c in df.columns if c not in _schema.HLA_META]:
            warnings.warn(
                f"{label}: column(s) {extra} are not part of the HLA schema and were dropped.",
                stacklevel=2,
            )
        # Require List columns up front. A plain-string locus column would otherwise be silently
        # wrapped into a 1-element list by the cast ("0201" -> ["0201"]), masking a notation
        # mistake — and set_hla does no parsing, so there is nothing to fall back on.
        non_list = [c for c in _schema.HLA_META if c != key and c in df.columns
                    and not isinstance(df.schema[c], pl.List)]
        if non_list:
            raise ValueError(
                f"{label}: HLA locus column(s) {non_list} must be List(str) of pre-parsed 4-digit "
                f"alleles, got {[str(df.schema[c]) for c in non_list]}. set_hla does not parse "
                f"notation."
            )
        if absent := [c for c in _schema.HLA_META if c != key and c not in df.columns]:
            df = df.with_columns([pl.lit(None, dtype=pl.List(pl.Utf8)).alias(c) for c in absent])

        self._ds.store(layout.HLA).write(df)              # the write's select+cast enforces HLA_META

    # --- shared validation ---------------------------------------------------------------

    def _validate_keys(self, df: pl.DataFrame, *, key: str, base_keys: List[str],
                       label: str) -> None:
        """Duplicate keys raise — the left join downstream would multiply rows. Keys the dataset
        does not have, and dataset entities the input does not mention, warn: both are usually
        typos, but both are legitimate when metadata arrives in pieces."""
        if dups := meta_edit.duplicate_keys(df, key):
            raise ValueError(f"{label}: duplicate {key} value(s) in input: {_fmt_ids(dups)}.")

        unknown, missing = meta_edit.check_keys(df.get_column(key).to_list(), base_keys)
        if unknown:
            warnings.warn(
                f"{label}: {len(unknown)} {key}(s) not present in the dataset "
                f"(rows kept but they will not surface in the join): {_fmt_ids(unknown)}.",
                stacklevel=3,
            )
        if missing:
            warnings.warn(
                f"{label}: {len(missing)} {key}(s) in the dataset have no row in the input: "
                f"{_fmt_ids(missing)}.",
                stacklevel=3,
            )
