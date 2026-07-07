"""Access to bundled read-only data files under tcr_io/resources/."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def resource_path(name: str) -> Path:
    """Absolute filesystem path to a bundled resource, e.g. 'hla_tcrs.parquet'."""
    return Path(str(files("tcr_io").joinpath("resources", name)))
