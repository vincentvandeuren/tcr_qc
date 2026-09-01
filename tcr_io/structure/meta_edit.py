"""Pure helpers for the metadata write path (`Dataset.set_*_meta` / `set_hla`).

These are deliberately side-effect-free set/frame operations: the `Dataset` methods do the
I/O (read old table, warn, `write_artifact`) and decide warn-vs-raise; everything decidable from
the frames alone lives here so it can be unit-tested without a dataset on disk.

The merge model is intentionally *additive* (see the design spec): a merge adds new columns and
new rows but never rewrites a column that already exists — the caller strips colliding columns
first (`reserved_column_collisions`), so `merge_frames` only ever brings in fresh columns.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import polars as pl


def duplicate_keys(df: pl.DataFrame, key: str) -> List[str]:
    """Values of `key` that appear more than once in `df` (each reported once)."""
    return (
        df.lazy()
        .select(key)
        .filter(pl.col(key).is_duplicated())
        .unique()
        .collect()
        .get_column(key)
        .drop_nulls()
        .to_list()
    )


def check_keys(new_keys: Iterable[str], base_keys: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split key membership relative to the dataset's base entities.

    Returns ``(unknown, missing)`` where ``unknown`` = keys in `new_keys` absent from the base
    (likely typos — the reader left-join would silently drop them) and ``missing`` = base keys
    absent from `new_keys` (entities left without metadata by this call). Both sorted."""
    new_set = set(new_keys)
    base_set = set(base_keys)
    unknown = sorted(new_set - base_set)
    missing = sorted(base_set - new_set)
    return unknown, missing


def reserved_column_collisions(new_cols: Iterable[str], reserved_cols: Iterable[str], key: str) -> List[str]:
    """Non-key columns of `new_cols` whose name already appears in `reserved_cols`.

    ``reserved_cols`` is the base table's columns (always) plus, in merge mode, the existing
    extra table's columns. A collision would either clash with a base column on the reader's
    join or fail to overwrite in an additive merge, so the caller warns and drops these."""
    reserved = set(reserved_cols) - {key}
    return [c for c in new_cols if c != key and c in reserved]


def merge_frames(old: pl.DataFrame, new: pl.DataFrame, key: str) -> pl.DataFrame:
    """Additively merge `new` into `old` on `key`.

    Full outer join, coalescing the key so no ``*_right`` column appears. `new` is expected to
    hold only `key` plus columns **not** already in `old` (the caller strips collisions), so the
    join adds new columns and new rows while leaving every existing `old` column untouched. Cells
    with no value after the join are null."""
    return old.join(new, on=key, how="full", coalesce=True)
