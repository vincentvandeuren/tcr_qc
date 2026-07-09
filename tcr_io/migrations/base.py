"""Migration registry.

Each dataset version bump is one migration that mutates a dataset in place and is
registered by the `@migration` decorator (keyed by `to_version`, so the chain is linear).
Migration functions operate on `ds.db_dir` + raw paths -- never the current-schema
accessors -- because they run on an *older* dataset than the library describes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tcr_io.dataset import TcrDataset   # type-only: migration modules never import dataset at runtime


@dataclass(frozen=True)
class Migration:
    to_version: int
    description: str
    fn: Callable[["TcrDataset"], None]


REGISTRY: dict[int, Migration] = {}   # to_version -> Migration


def migration(to_version: int, description: str):
    def register(fn):
        if to_version in REGISTRY:
            raise ValueError(f"Duplicate migration to v{to_version}: {fn.__name__}")
        REGISTRY[to_version] = Migration(to_version, description, fn)
        return fn
    return register
