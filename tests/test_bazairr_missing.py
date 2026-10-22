"""Without the private `bazairr` package, the index operation and `ds.query` say so."""
import sys
import warnings

import pytest

from tcr_io import Dataset
from tcr_io.operations import ConstructBazairrIndex

from test_operations import _make_dataset


def test_bazairr_missing_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "bazairr", None)
    warnings.simplefilter("ignore")
    ds = Dataset(_make_dataset())
    with pytest.raises(ImportError, match="bazairr"):
        ConstructBazairrIndex()
    with pytest.raises(ImportError, match="bazairr"):
        ds.query
