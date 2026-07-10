"""v2 -> v3: fold hive outputs into unstructured dirs; drop the legacy ``temp/`` scratch.

The unstructured-output change (docs/unstructured_output_plan.md) removed
``HiveOutput`` / ``Kind.HIVE``: ``TabulateByVJ`` now writes its ``(v_gene, j_gene)``
partitions into an *unstructured* directory read back as a ``Path``, and
``OverlapAnalyzer`` writes its per-repertoire "heads" into a managed ``operations/`` dir
instead of a hardcoded ``temp/heads``.

This migration deletes:
  - the ``tabulate_by_vj_gene`` op dir — its ``operation.json`` still records ``kind: "hive"``,
    which the current reader no longer understands; it regenerates in the new shape on demand;
  - any legacy top-level ``temp/`` scratch left by the old overlap analyzer.

Both are generated / rebuildable, so deletion is safe (clear & regenerate, as for v2).
"""
import shutil

from .base import migration
from ..layout import Layout


def _rm(path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@migration(to_version=3, description="Fold hive outputs into unstructured dirs; drop legacy temp/")
def upgrade(ds):
    _rm(ds.db_dir / Layout.operations_dir.path / "tabulate_by_vj_gene")
    _rm(ds.db_dir / "temp")
