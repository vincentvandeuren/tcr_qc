"""Finding near-identical junctions in bulk.

Thin wrapper over the Rust ``hd1_pairs``, which is itself a thin wrapper over
the ``symscan`` crate (symmetric deletion lookup). The Rust side returns row
positions because it has no opinion about how rows are identified; this module
turns those into whatever column the caller actually uses, and documents the
conventions that are easy to get wrong.

The Rust function is reached through ``tcr_io._internal``, not through
``register_plugin_function``: a polars expression maps N rows to N rows, and an
edge list does not have that shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import polars as pl

from tcr_io._internal import hd1_pairs as _hd1_pairs
from tcr_io._internal import neighbor_counts as _neighbor_counts

__all__ = ["count_similar", "find_similar_pairs", "stream_similar_pairs"]


def find_similar_pairs(
    df: pl.DataFrame,
    *,
    sequence: str = "junction_aa",
    by: Optional[Sequence[str]] = ("v_call",),
    max_distance: int = 1,
    metric: str = "hamming",
    id_column: Optional[str] = None,
) -> pl.DataFrame:
    """Every pair of rows whose sequences are within ``max_distance``.

    Args:
        df: Frame holding the sequences. Rows with a null ``sequence`` are
            skipped; they cannot be anyone's neighbour.
        sequence: Column to compare.
        by: Columns a pair must agree on. Defaults to ``("v_call",)``.

            This is not a convenience. On real TRB data **85% of Hamming-1
            junction pairs join rows with different ``v_call``** -- ``CASS…``
            is produced by dozens of TRBV genes, so V identity is not
            determined within the junction. Searching without ``by`` therefore
            returns a very different, much larger relation.

            Note what is *not* here: ``j_call``. Requiring it as well drops a
            further ~7% of same-V pairs, because two genuine neighbours can
            carry different J calls -- TRBJ2-3 trims to ``DTQYF`` and TRBJ2-5 to
            ``ETQYF``, one substitution apart. Add it only if you also intend
            whatever you compare against to condition on J.

            Pass ``None`` to search the whole frame as one group.
        max_distance: Maximum edit distance. ``1`` finds identical and
            one-edit-apart pairs.
        metric: ``"hamming"`` (substitutions only, equal lengths) or
            ``"levenshtein"`` (also indels). Hamming is the default because an
            indel changes junction length and is a different event from a
            substitution.
        id_column: When given, ``idx_a``/``idx_b`` are replaced by
            ``<id_column>_a``/``<id_column>_b`` carrying that column's values.
            Otherwise row positions into ``df`` are returned.

    Returns:
        One row per pair, with ``distance``. Each pair appears **once**.

        Ordering is by row position (``idx_a < idx_b``), which survives the
        mapping to ``id_column`` as a stable convention but is *not* a sort of
        the identifier values -- row 9 precedes row 10, whereas the strings
        ``"10"`` and ``"9"`` sort the other way. Compare pair sets as unordered
        pairs if you are checking this against another implementation.

        ``distance == 0`` is included and is not a bug. When rows are keyed by
        nucleotide clonotype, two rows sharing a ``junction_aa`` are
        independent recombination events that converged on one amino-acid
        sequence -- real observations, not duplicates. Filter on ``distance``
        if you want only substitutions.

    Raises:
        ValueError: If ``sequence`` contains non-ASCII data. symscan rejects
            the whole input rather than skipping offending rows, so filter
            upstream if that is possible in your data.
    """
    _validate(df, sequence, by, max_distance, metric)
    by_cols = [] if by is None else list(by)
    pairs = _hd1_pairs(df, sequence, by_cols, max_distance, metric)
    if id_column is None:
        return pairs

    return _with_ids(df, pairs, id_column)


def _validate(df, sequence, by, max_distance, metric) -> None:
    if metric not in ("hamming", "levenshtein"):
        raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    by_cols = [] if by is None else list(by)
    missing = [c for c in [sequence, *by_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"column(s) not in frame: {missing}")


def _with_ids(df: pl.DataFrame, pairs: pl.DataFrame, id_column: str) -> pl.DataFrame:
    """Replace row positions with values from ``id_column``.

    Gathers on the Series directly rather than inside a ``select()``: an
    expression would resolve ``idx_a`` against the id column's own frame, where
    it does not exist.
    """
    if id_column not in df.columns:
        raise ValueError(f"column(s) not in frame: [{id_column!r}]")
    ids = df.get_column(id_column)
    return pl.DataFrame(
        {
            f"{id_column}_a": ids.gather(pairs.get_column("idx_a")),
            f"{id_column}_b": ids.gather(pairs.get_column("idx_b")),
            "distance": pairs.get_column("distance"),
        }
    )


def count_similar(
    df: pl.DataFrame,
    *,
    sequence: str = "junction_aa",
    by: Optional[Sequence[str]] = ("v_call",),
    max_distance: int = 1,
    metric: str = "hamming",
) -> pl.DataFrame:
    """How many neighbours each row has, without building the edge list.

    Prefer this to aggregating :func:`find_similar_pairs` whenever the degree
    is what you actually want. Going via the pairs materialises every edge only
    to collapse it back to one number per row, and the edge count grows much
    faster than the row count -- so the intermediate is the largest thing in
    the pipeline and it is pure waste. Here each group's pairs are consumed and
    dropped as they are produced.

    Args:
        df, sequence, by, max_distance, metric: As :func:`find_similar_pairs`.

    Returns:
        A frame aligned to ``df``, one row each, with:

        ``n_neighbors``
            Rows within ``max_distance``, excluding the row itself.
        ``n_exact``
            The subset at distance 0: other rows whose sequence is
            byte-identical. When rows are keyed by nucleotide clonotype these
            are independent rearrangements that converged on the same
            amino-acid sequence -- convergent recombination, driven by how
            easy the sequence is to generate. That is a different event from a
            relative one substitution away, which is the motif-family signal an
            enrichment test is usually hunting for, so the two are kept apart.
            Collapsing them cannot be undone without rebuilding the edge list
            this function exists to avoid.

            ``n_neighbors - n_exact`` gives everything at distance >= 1.
            Note that with ``max_distance > 1`` this lumps all non-zero
            distances together -- ``n_exact`` is only ever the zero bucket. For
            a per-distance breakdown use :func:`find_similar_pairs` and group
            on ``distance``.

    Example:
        >>> counts = count_similar(df, by=["v_call"])            # doctest: +SKIP
        >>> df.hstack(counts).filter(pl.col("n_neighbors") > 2)  # doctest: +SKIP
    """
    _validate(df, sequence, by, max_distance, metric)
    by_cols = [] if by is None else list(by)
    return _neighbor_counts(df, sequence, by_cols, max_distance, metric)


def stream_similar_pairs(
    df: pl.DataFrame,
    sink_dir: "str | Path",
    *,
    sequence: str = "junction_aa",
    by: Optional[Sequence[str]] = ("v_call",),
    max_distance: int = 1,
    metric: str = "hamming",
    id_column: Optional[str] = None,
) -> pl.LazyFrame:
    """Write pairs group-by-group to parquet and return a scan over them.

    For when you want the edges but cannot hold them. Peak memory is one
    group's pairs rather than the whole cohort's, and the returned
    :class:`~polars.LazyFrame` lets polars stream the aggregation.

    Grouping happens here rather than in Rust so that each group's result can
    be written and released before the next is searched; the Rust function is
    called per group with ``by`` already applied.

    Args:
        sink_dir: Directory for the parquet parts. Created if absent. Existing
            ``*.parquet`` files in it are removed first, so a re-run does not
            silently read stale parts alongside new ones.
        df, sequence, by, max_distance, metric, id_column:
            As :func:`find_similar_pairs`.

    Returns:
        A ``LazyFrame`` over the written parts.

    Example:
        >>> edges = stream_similar_pairs(df, "edges/", id_column="clonotype_id")
        ... # doctest: +SKIP
        >>> edges.group_by("clonotype_id_a").len().collect(engine="streaming")
        ... # doctest: +SKIP
    """
    from pathlib import Path as _Path

    _validate(df, sequence, by, max_distance, metric)
    sink = _Path(sink_dir)
    sink.mkdir(parents=True, exist_ok=True)
    for stale in sink.glob("*.parquet"):
        stale.unlink()

    by_cols = [] if by is None else list(by)
    parts = df.group_by(by_cols) if by_cols else [((), df)]
    for n, (_, group) in enumerate(parts):
        pairs = _hd1_pairs(group, sequence, [], max_distance, metric)
        if id_column is not None:
            pairs = _with_ids(group, pairs, id_column)
        if pairs.height:
            pairs.write_parquet(sink / f"part-{n:05d}.parquet")
    return pl.scan_parquet(sink / "*.parquet")
