"""v1 -> v2: move to the ``operations/`` dir; drop scattered op outputs + the global ledger.

This is a **clear & regenerate** migration (breaking change). Old operation outputs were
scattered under ``qc/``, ``tabulated/`` and ``meta/{repertoire,patient,publication}/``, with a
global ``meta/operations.json`` ledger. v2 puts every operation output under
``operations/<op>/[<LOCUS>/]`` with a self-describing per-op ``operation.json``.

Rather than relocate/parse old results we simply delete them and recreate an empty
``operations/`` dir — the ops re-run on demand into the new layout. Canonical dataset tables
(``repertoire.parquet``, ``patient.parquet``, ``hla.parquet``, the ``*_meta.parquet`` side
files) are left untouched.
"""
import shutil
from pathlib import Path

from .base import migration

_V2_OPERATIONS_DIR = "operations"        # pinned; not Layout.operations_dir

# Op outputs that historically landed under meta/ (must be enumerated so we don't touch the
# canonical tables living in the same dirs).
_SCATTERED_OP_FILES = [
    "meta/operations.json",                                  # the old global ledger
    "meta/patient/inferred_hla.parquet",
    "meta/patient/inferred_hla_long.parquet",
    "meta/repertoire/inferred_hla.parquet",
    "meta/repertoire/inferred_hla_long.parquet",
    "meta/repertoire/inferred_hla_distance.parquet",
    "meta/repertoire/gene_counts.parquet",
    "meta/repertoire/v_counts.parquet",
    "meta/repertoire/j_counts.parquet",
    "meta/repertoire/vj_bias.parquet",
    "meta/repertoire/vj_long.parquet",
    "meta/repertoire/diversity_summary.parquet",
    "meta/repertoire/vdj_statistics_summary.parquet",
    "meta/repertoire/ecocluster_hits.parquet",
    "meta/repertoire/mait_hits.parquet",
    "meta/publication/publication_metadata.json",
]

# Whole generated roots that only ever held op outputs.
_SCATTERED_OP_DIRS = ["qc", "tabulated"]


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@migration(to_version=2, description="Move to operations/ dir; drop scattered op outputs + ledger")
def upgrade(db_dir: Path) -> None:
    for rel in _SCATTERED_OP_DIRS + _SCATTERED_OP_FILES:
        _rm(db_dir / rel)
    # Nothing is rebuilt here — ops re-run on demand into operations/.
    (db_dir / _V2_OPERATIONS_DIR).mkdir(exist_ok=True)
