"""Dataset **format contract**: what tables exist (`schema`), where they live
(`layout`), what version the format is (`version`), and how to move between versions
(`migrations`). Everything else in `tcr_io` is behaviour over this contract.

Common names are re-exported here, so callers can `from tcr_io.structure import Layout,
REPERTOIRE_META, DATASET_VERSION`. The `migrations` subpackage is imported lazily by its
users (not here) to keep the pkgutil auto-import off the `import tcr_io` path.
"""
from . import schema
from . import meta_edit
from .schema import (
    GENERATION_META,
    REPERTOIRE,
    REPERTOIRE_META,
    PATIENT_META,
    PUBLICATION_META,
    CLONE_TO_CELL,
    HLA_META,
)
from .layout import (
    Artifact,
    Kind,
    Role,
    Layout,
    LAYOUT,
    REQUIRED_DIRS,
    GENERATED_DIRS,
    repertoire_dir_relpath,
    repertoire_locus_relpath,
    clone_to_cell_relpath,
    loci_glob,
    repertoire_meta_relpath,
    operation_relpath,
    safe_repertoire_name,
    render_tree,
)
from .writers import write_artifact
from .version import DATASET_VERSION, Manifest

__all__ = [
    "schema", "meta_edit",
    "GENERATION_META", "REPERTOIRE", "REPERTOIRE_META", "PATIENT_META",
    "PUBLICATION_META", "CLONE_TO_CELL", "HLA_META",
    "Artifact", "Kind", "Role", "Layout", "LAYOUT",
    "REQUIRED_DIRS", "GENERATED_DIRS",
    "repertoire_dir_relpath", "repertoire_locus_relpath", "clone_to_cell_relpath",
    "loci_glob", "repertoire_meta_relpath",
    "operation_relpath", "safe_repertoire_name", "render_tree",
    "write_artifact",
    "DATASET_VERSION", "Manifest",
]
