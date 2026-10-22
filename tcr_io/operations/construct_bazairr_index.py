"""`ConstructBazairrIndex`: one `.bazairr` CDR3 similarity index per V gene of a dataset's passing clonotypes. The
index itself is built by the private `bazairr` package, which is imported when the operation is created or run."""
from __future__ import annotations

from dataclasses import dataclass

from ..structure import Artifact, Format, Store
from .base import BaseOperation

MISSING = "ConstructBazairrIndex requires the bazairr package (a private companion of tcrio), which is not installed"


def import_bazairr():
    """the `bazairr` module; ImportError naming it if it cannot be imported"""
    try:
        import bazairr
    except ImportError as e:
        raise ImportError(MISSING) from e
    return bazairr


@dataclass
class ConstructBazairrIndex(BaseOperation):
    name = "construct_bazairr_index"
    version = "2.0"
    description = "One .bazairr CDR3 similarity index per V gene of the passing clonotypes (TRB only)."
    supported_loci = frozenset({"TRB"})

    index = Artifact("index", Format.BAZAIRR_DIR)

    def __post_init__(self) -> None:
        import_bazairr()

    def _run(self, ds, out: Store) -> None:
        import_bazairr().build_index(ds, out(self.index).clear())
