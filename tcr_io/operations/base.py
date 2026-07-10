from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
import traceback
from typing import ClassVar, Dict, List, Optional, Union, TYPE_CHECKING
import polars as pl

if TYPE_CHECKING:
    from tcr_io.dataset import TcrDataset


# Sentinel for `supported_loci`: run the op once per locus present in the dataset.
ALL_LOCI = object()

# A single operation output value the framework serialises (parquet/ndjson/json).
# Unstructured *directory* outputs are NOT values: the op writes them itself into the managed
# dir handed out by `TcrDataset._operation_output_dir` and they never appear here (see
# docs/unstructured_output_plan.md).
Result = Union[pl.DataFrame, pl.LazyFrame, list]


@dataclass
class OperationResults:
    """Outputs of one `_run`, keyed by output **name** (not path).

    The framework roots each output at ``operations/<op>/[<locus>/]<name>.<ext>`` and
    writes it by `Kind`.
    """
    outputs: Dict[str, Result]


@dataclass
class OperationFailure:
    error: str


@dataclass
class OutputRecord:
    """One row of `OperationRecord.outputs` — self-describing so retrieval is deterministic."""
    name: str
    path: str                   # relative to the op dir, e.g. "TRB/gene_counts.parquet"
    kind: str                   # Kind value: "parquet" | "ndjson" | "json" | "unstructured"
    locus: Optional[str]        # None for locus-agnostic outputs

    @classmethod
    def from_dict(cls, d: dict) -> "OutputRecord":
        return cls(name=d["name"], path=d["path"], kind=d["kind"], locus=d.get("locus"))


@dataclass
class OperationRecord:
    """Serialised to ``operations/<name>/operation.json`` — the per-op ledger.

    Replaces the old global ``meta/operations.json``. Source of truth for skip logic,
    result retrieval and `ds.rerun`.
    """
    operation_name: str
    version: str
    description: str
    ran_at: datetime
    duration_s: float
    status: str                     # "success" | "failure"
    error: Optional[str]
    params: dict                    # config this op ran with -> rerun defaults
    loci: Optional[List[str]]       # which loci ran (None for locus-agnostic ops)
    outputs: List[OutputRecord]

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
            loci=d.get("loci"),
            outputs=[OutputRecord.from_dict(o) for o in d.get("outputs", [])],
        )


class BaseOperation(ABC):
    """A generated-result computation over a `TcrDataset`.

    Concrete ops are `@dataclass`es whose **annotated** fields are config; the metadata
    attributes below stay **unannotated** on purpose so the dataclass never treats them as
    fields (annotate config, leave metadata bare — see base_operation_design.md §2).
    Resources (models, reference frames) load in `__post_init__` as plain attributes → not
    fields → excluded from `params()`.
    """
    name = "base_operation"
    version = "0.0"
    description = "Base operation - does nothing"
    # None -> locus-agnostic (one pass); ALL_LOCI -> per present locus; frozenset({"TRB"}) -> subset
    supported_loci: ClassVar[Union[frozenset, object, None]] = None

    @abstractmethod
    def _run(self, ds: "TcrDataset", locus: Optional[str] = None) -> OperationResults:
        """Compute outputs for one locus (or the whole dataset when locus is None),
        returned keyed by *name*; the framework assigns paths + writes them."""

    def params(self) -> dict:
        """Serialisable config this op ran with (dataclass fields). Empty for non-dataclass ops."""
        if not is_dataclass(self):
            return {}
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def loci_to_run(self, present: frozenset) -> List[Optional[str]]:
        if self.supported_loci is None:
            return [None]
        if self.supported_loci is ALL_LOCI:
            return sorted(present)
        return sorted(self.supported_loci & present)

    def run(self, ds: "TcrDataset", locus: Optional[str] = None) -> Union[OperationResults, OperationFailure]:
        try:
            return self._run(ds, locus)
        except Exception as e:
            return OperationFailure(error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


@dataclass
class TestNullOperation(BaseOperation):
    name = "test_null_operation"
    version = "0.1"
    description = "Test operation that creates some empty output"

    def _run(self, ds: "TcrDataset", locus: Optional[str] = None) -> OperationResults:
        filter_pass = ds.repertoire_meta.with_columns(
            filter_pass_pct=pl.col("n_clonotypes") / pl.col("n_clonotypes").add(pl.col("n_filtered_clonotypes"))
        ).select(["repertoire_id", "filter_pass_pct"])

        return OperationResults(outputs={"filter_pass": filter_pass})
