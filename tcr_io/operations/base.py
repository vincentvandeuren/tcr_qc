"""`BaseOperation` — what an operation IS.

An operation declares its outputs as `Artifact` class attributes and computes them. It does
not decide where it writes, whether it needs to run, or what happens when it raises: that is
all `OperationRunner`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import ClassVar, List, TYPE_CHECKING

import polars as pl

from ..expressions import KNOWN_LOCI
from ..structure import Artifact, Format, Store
from .record import serialize_params

if TYPE_CHECKING:
    from tcr_io.dataset import Dataset


# `supported_loci` is a plain frozenset intersected with what the runner offers — no sentinel,
# and no `None`. "Every locus" is spelled `KNOWN_LOCI`, and unlike a sentinel it can be unioned
# with `UNASSIGNED`, which `FilteringReport` needs. The `None` third state meant "locus-agnostic,
# one pass"; no shipped operation is, since publication fetching became a metadata writer, so it
# bought a branch in the runner, in `loci_to_run`, in `Store.with_root` and in every reader of a
# record's `locus` — for a case that does not exist.


class BaseOperation(ABC):
    """A generated-result computation over a `Dataset`.

    Concrete ops are `@dataclass`es whose **annotated** fields are config; the metadata
    attributes below stay **unannotated** on purpose so the dataclass never treats them as
    fields. Resources (models, reference frames) load in `__post_init__` as plain attributes
    -> not fields -> excluded from `params()`.

    Outputs are `Artifact` class attributes. That is the whole output declaration: it gives
    each output a name (the attribute), a path shape, a format and a schema in one place, and
    it makes a typo an `AttributeError` at the write site instead of a lookup failure at the
    read site three weeks later.
    """
    name = "base_operation"
    version = "0.0"
    description = "Base operation - does nothing"
    # Intersected with the loci the dataset offers. The default runs on every real locus;
    # narrow it (`frozenset({"TRB"})`) or widen it (`KNOWN_LOCI | {UNASSIGNED}`) per op.
    supported_loci: ClassVar[frozenset] = KNOWN_LOCI

    @abstractmethod
    def _run(self, ds: "Dataset", out: Store) -> None:
        """Compute and write this pass's outputs.

        `ds` is already bound to the pass's locus (or unbound for a locus-agnostic op), so
        every locus-grain read inside needs no locus argument. `out` is a Store rooted at
        this pass's output directory: `out(SOME_ARTIFACT).write(frame)`, or for a directory
        artifact `out(SOME_DIR).clear()` and write into the returned path.

        Writing is the op's own act rather than a value it returns, so a table output and a
        directory output are produced the same way — the old `_pending_unstructured` list
        existed only because they were not. The runner discovers what was written by globbing
        the declared artifacts afterwards.

        RAISES on failure. There is no `run()` wrapper and no `OperationFailure`: those were
        a wrapper class, a union return type and a branch, all to move a try/except down one
        level. The runner catches.
        """

    @classmethod
    def artifacts(cls) -> tuple:
        """Every output this operation declares.

        Walks `reversed(cls.__mro__)` so a subclass override wins, then insists that each
        surviving artifact was declared by `cls` itself. Outputs belong to the class that
        RUNS: an inherited artifact would carry its base class's `owner`, and `ds.result()`
        resolves an output's directory from `owner.name` — so a shared declaration would send
        every subclass's results to the base class's folder.
        """
        found = {}
        for klass in reversed(cls.__mro__):
            found.update({k: v for k, v in vars(klass).items() if isinstance(v, Artifact)})
        stolen = [a for a in found.values() if a.owner is not cls]
        if stolen:
            raise TypeError(
                f"{cls.__name__} inherits output(s) "
                f"{[a.key for a in stolen]} from {stolen[0].owner.__name__}; declare them on "
                f"{cls.__name__} so they are written and read under {cls.name!r}."
            )
        return tuple(found.values())

    def params(self) -> dict:
        """The config this ran with — rebuild defaults, and part of run identity. Empty for a
        non-dataclass op. Serialized here so the value a record stores and the value it is
        compared against are the same value."""
        if not is_dataclass(self):
            return {}
        return serialize_params({f.name: getattr(self, f.name) for f in fields(self)})

    def loci_to_run(self, offered: frozenset) -> List[str]:
        """This op's loci that the dataset actually offers — `offered` is `ds.dispatch_loci`.

        Not an override point: fan-out is runner policy, and with `_unassigned` expressible in
        `supported_loci` there is no longer a reason to want one.
        """
        return sorted(self.supported_loci & offered)


@dataclass
class NullOperation(BaseOperation):
    """The smallest complete operation: one table, no config.

    Lives here rather than in the tests because it is the worked example of the contract —
    declare an Artifact, write through `out`, return nothing.
    """
    name = "null_operation"
    version = "0.2"
    description = "Per-repertoire fraction of clonotypes that passed ingest filtering."

    pass_rate = Artifact("pass_rate.parquet", Format.PARQUET)

    def _run(self, ds: "Dataset", out: Store) -> None:
        out(self.pass_rate).write(
            ds.repertoire_counts
            .with_columns(pass_pct=pl.col("n_clonotypes")
                          / pl.col("n_clonotypes").add(pl.col("n_filtered_clonotypes")))
            .select(["repertoire_id", "pass_pct"])
        )
