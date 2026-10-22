"""v7 -> v8: drop the `TabulateByVJ` results.

`TabulateByVJ` is retired in favour of the CDR3 index (`ConstructBazairrIndex`). Its output
is derived data, so the operation directory is deleted rather than migrated: nothing reads it
any more, and no backup is needed.

At v7 operation outputs live at ``operations/{operation_name}/{locus}/`` (v6 cleared
``operations/`` and fixed that shape), so the whole ``operations/tabulate_by_vj_gene`` tree,
every locus under it, goes in one removal.

Every path below is a v7-era literal. None may come from the live layout. See base.py.
"""
import shutil
from pathlib import Path

from .base import migration

_V7_OPERATIONS_DIR = "operations"        # pinned; not Layout.operations_dir
_TABULATION = "tabulate_by_vj_gene"


@migration(to_version=8, description="Drop the retired TabulateByVJ results")
def upgrade(db_dir: Path) -> None:
    shutil.rmtree(db_dir / _V7_OPERATIONS_DIR / _TABULATION, ignore_errors=True)
