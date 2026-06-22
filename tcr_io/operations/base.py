from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Dict, Union, TYPE_CHECKING, Optional
from pathlib import Path
import polars as pl
from time import time
import traceback
if TYPE_CHECKING:
    from tcr_io.dataset import TcrDataset



@dataclass
class OperationResults:
    outputs: Dict[str, Union[pl.LazyFrame, pl.DataFrame, dict]] # relative path to dataset dir
    duration_s: Optional[float] = None

@dataclass
class OperationFailure:
    error: str
    duration_s: Optional[float] = None

@dataclass
class OperationMeta:
    operation_name: str
    version: str
    ran_at: pl.Datetime
    duration_s: float
    status: Literal["success", "failure"]
    error: Optional[str]
    description: str
    outputs: List[str]

class BaseOperation(ABC):
    name = "base_operation"
    version = "0.0"
    description = "Base operation - does nothing"

    @abstractmethod
    def _run(self, ds:TcrDataset) -> OperationResults:
        pass

    def run(self, ds:TcrDataset) -> OperationResults | OperationFailure:
        s = time()
        try:
            results = self._run(ds)
        except Exception as e:
            results = OperationFailure(
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
        results.duration_s = time() - s
        return results
    


class TestNullOperation(BaseOperation):
    name = "test_null_operation"
    version = "0.1"
    description = "Test operation that creates some empty output"

    def _run(self, ds:TcrDataset) -> OperationResults:
        filter_pass = ds.repertoire_meta.with_columns(
            filter_pass_pct = pl.col("n_clonotypes") / pl.col("n_clonotypes").add(pl.col("n_filtered_clonotypes"))
        ).select(["repertoire_id", "filter_pass_pct"])

        return OperationResults(
            outputs={"qc/filter_pass.parquet": filter_pass}
        )