"""Migration registry, and the walker over it.

A migration runs on an **older tree than the current layout describes**. It is therefore
given a bare ``db_dir`` path and nothing else: no ``Dataset``, no layout constant, no
schema. Every path, glob and column list it needs is pinned as a literal in its own
module, frozen at the version it upgrades *to*.

This is not stylistic. A migration that imports a live symbol silently changes meaning
whenever that symbol moves — and it changes meaning *in the past*, on trees that were
written before the move. The failure is silent by construction: a stale glob matches
nothing, the step no-ops, and the next migration runs on data that never got upgraded.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..layout import MANIFEST
from ..store import Store
from ..version import DATASET_VERSION, Manifest

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    to_version: int
    description: str
    fn: Callable[[Path], None]      # takes db_dir; never a Dataset


REGISTRY: dict[int, Migration] = {}


def migration(to_version: int, description: str):
    """Register ``fn`` as the v(to_version - 1) -> v(to_version) step."""
    def register(fn):
        if to_version in REGISTRY:
            raise ValueError(f"Duplicate migration to v{to_version}: {fn.__name__}")
        REGISTRY[to_version] = Migration(to_version, description, fn)
        return fn
    return register


class Migrator:
    """Walks a dataset from the version on disk up to `target`, one registered step at a time.

    A writer, peer to `DatasetIngester`: it holds a `Store` and mutates a directory. Separate
    from the read class because migration is the one operation that runs against a tree the
    current layout does *not* describe — see this module's docstring.

    The commit is `update(version=...)`, not a whole-value write. Since v7 the manifest holds
    `version` alone — ingest provenance moved to `meta/generation.json` — so this is no longer
    about split ownership. It is about the future: a walk that reconstructed the manifest from
    the fields THIS version knows would silently drop any key a later version adds, on exactly
    the trees a migration exists to carry forward.
    """

    def __init__(self, db_dir: str | Path):
        self.db_dir = Path(db_dir)
        if not self.db_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.db_dir}")
        self.store = Store(self.db_dir)

    @property
    def version(self) -> int:
        """Re-read every time. A migration commits after each step, so a cached value would
        be stale for the rest of the walk."""
        return Manifest.read(self.store).version

    def pending(self, target: int = DATASET_VERSION) -> list[Migration]:
        """The chain from the on-disk version to `target`. Raises rather than skipping a gap:
        a missing link means the data would be left in a shape no version describes."""
        plan, v = [], self.version
        while v < target:
            step = REGISTRY.get(v + 1)
            if step is None:
                raise RuntimeError(f"No migration registered for v{v} -> v{v + 1}")
            plan.append(step)
            v = step.to_version
        return plan

    def migrate(self, target: int = DATASET_VERSION, *, dry_run: bool = False) -> list[Migration]:
        """Apply `pending()`, committing the version after each step so a mid-chain failure
        leaves a resumable state rather than a tree that claims a version it does not have."""
        plan = self.pending(target)
        for step in plan:
            log.info("migrate v%d -> v%d: %s",
                     step.to_version - 1, step.to_version, step.description)
            if not dry_run:
                step.fn(self.db_dir)                            # a bare path, never a Dataset
                self.store(MANIFEST).update(version=step.to_version)
        return plan
