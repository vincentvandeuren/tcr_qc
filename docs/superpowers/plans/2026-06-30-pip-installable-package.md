# Pip-Installable Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `tcrio` maturin/PyO3 package cleanly `pip install`-able and usable, ready for a closed beta via GitHub Release wheels.

**Architecture:** Declare runtime dependencies and package metadata in `pyproject.toml`; bundle the HLA + CMV model data as pre-processed parquet under `tcr_io/resources/` (dropping the pickle/TSV loading); populate the public API; and land the minimum fixes that turn the red CI green. The bundled parquets and the converter script (`scripts/build_resources.py`, `model_sources/`) already exist in the working tree from a prior session — this plan commits and consumes them.

**Tech Stack:** Rust (PyO3, pyo3-polars, polars 0.53), Python 3.10+, polars, maturin, pytest, ruff, mypy.

**Design spec:** `docs/superpowers/specs/2026-06-30-pip-installable-package-design.md`

## Global Constraints

- Distribution name `tcrio`; import package `tcr_io`. Do not rename either.
- Python: `requires-python = ">=3.10"`, Rust build `abi3-py310`. 3.10 is the true floor (polars ≥3.10, `match/case`, PEP 604 unions). Tested on 3.10–3.14.
- Runtime deps use `>=` lower bounds in `pyproject.toml`, **no upper caps except `polars`** — polars is a compiled-plugin ABI boundary and is capped `<2.0` (see Task 3).
- `requirements.txt` stays the exact-pinned (`==`) dev/CI lockfile — not the source of truth for installs.
- Everything is a core dependency for now (no `[viz]` split); dev tooling lives in `[project.optional-dependencies].dev`.
- Bundled resources are read-only parquet in `tcr_io/resources/`; access them via `importlib.resources`, never a hardcoded `/data/...` path.
- The CI `-Dwarnings` gate is **removed** (Option A): the Rust crate is under active development with intentional not-yet-wired-up code, so only unused *imports* are cleaned, not dead functions/structs. `cargo clippy` still runs in `make pre-commit` for local visibility. Already applied (commit `fafa85d`).
- Version comes from `Cargo.toml` via `dynamic = ["version"]`. Cargo needs SemVer `0.1.0-beta.1`; maturin normalizes it to the PEP 440 wheel version `0.1.0b1`, while Python `__version__` (from `CARGO_PKG_VERSION`) reads the raw `0.1.0-beta.1`.
- Verify commands assume a venv where the extension is built (`maturin develop`) and dev deps are installed.

---

## File Structure

- `Cargo.toml` — drop the unused `rust-stemmers` dependency.
- `src/expressions.rs` — remove four unused `use` imports.
- `src/reference_points.rs` — remove the `use std::usize;` import.
- `pyproject.toml` — dependencies, `[dev]` extra, metadata, `[tool.maturin] include`, `requires-python`.
- `tcr_io/_resources.py` (new) — one helper: `resource_path(name) -> Path`.
- `tcr_io/operations/hla.py` — read bundled parquets; `model_checkpoint=None` default.
- `tcr_io/operations/cmv_hits.py` — read bundled parquet; `model_checkpoint=None` default.
- `tcr_io/operations/unconventional_tcrs.py` — `MaitHits` requires an explicit path (deferred, not bundled).
- `tcr_io/operations/metadata.py` — delete the dead commented block containing the leaked API key.
- `tcr_io/__init__.py` — public API surface + `__version__`.
- `tests/test_pig_latinnify.py` — delete (broken tutorial leftover).
- `tests/test_smoke.py` (new) — import + version.
- `tests/test_resources.py` (new) — bundled ops load; MAIT raises.
- `README.md` — real beta README.
- `Cargo.toml` version — bump to `0.1.0-beta.1` (SemVer; maturin → wheel `0.1.0b1`).

Already on disk from a prior session (this plan just commits them): `tcr_io/resources/*.parquet`, `scripts/build_resources.py`, `model_sources/README.md`, updated `.gitignore`.

---

### Task 1: Make the Rust crate warning-clean and build the extension

**Files:**
- Modify: `src/expressions.rs:4-7`
- Modify: `src/reference_points.rs:1`
- Modify: `Cargo.toml` (dependencies section)

**Interfaces:**
- Produces: a compiled `tcr_io._internal` extension (so all later Python imports of `tcr_io.expressions` work), and the crate's unused *imports* removed. (Full clippy-clean is NOT a goal — the `-Dwarnings` CI gate was removed per Option A; see Global Constraints.)

- [ ] **Step 1: Remove the four unused imports in `src/expressions.rs`**

Delete these four lines (currently lines 4-7):

```rust
use num_traits::Signed;
use polars::prelude::arity::broadcast_binary_elementwise;
use rust_stemmers::{Algorithm, Stemmer};
use std::fmt::Write;
```

Leave the remaining `use` lines (`polars::prelude::*`, `pyo3_polars::derive::polars_expr`, `serde::Deserialize`, and the three `crate::` imports) untouched.

- [ ] **Step 2: Remove the deprecated import in `src/reference_points.rs`**

Delete line 1:

```rust
use std::usize;
```

- [ ] **Step 3: Remove the unused `rust-stemmers` dependency in `Cargo.toml`**

Delete this line from `[dependencies]`:

```toml
rust-stemmers = "1.2.0"
```

(Leave `num-traits` — removing the import is enough; an unused crate dep is not a warning.)

- [ ] **Step 4: Verify the crate is warning-clean**

Run: `RUSTFLAGS="-Dwarnings" cargo clippy --all-features`
Expected: finishes with no errors/warnings (exit 0). If clippy reports any other unused import, remove exactly that import and re-run until clean.

- [ ] **Step 5: Build the extension into the current venv**

Run: `maturin develop`
Expected: `🛠 Installed tcrio-0.1.0` (or similar) and exit 0.

- [ ] **Step 6: Confirm the extension imports and `to_imgt` works**

Run: `python -c "import polars as pl; from tcr_io.expressions import to_imgt; print(pl.DataFrame({'v':['TCRBV05-06']}).with_columns(to_imgt(pl.col('v')).alias('i'))['i'][0])"`
Expected: `TRBV5-6*01`

- [ ] **Step 7: Commit**

```bash
git add src/expressions.rs src/reference_points.rs Cargo.toml
git commit -m "fix: remove unused Rust imports and rust-stemmers dep (unblocks -Dwarnings CI)"
```

---

### Task 2: Bundled-resource accessor + commit the generated resources

**Files:**
- Create: `tcr_io/_resources.py`
- Test: `tests/test_resources_helper.py`
- Commit (already on disk): `tcr_io/resources/hla_tcrs.parquet`, `tcr_io/resources/hla_model_weights.parquet`, `tcr_io/resources/cmv_ecocluster.parquet`, `scripts/build_resources.py`, `model_sources/README.md`, `.gitignore`

**Interfaces:**
- Produces: `tcr_io._resources.resource_path(name: str) -> pathlib.Path` — absolute path to a file in `tcr_io/resources/`. Used by Tasks 4 and 5.

- [ ] **Step 1: Write the failing test**

Create `tests/test_resources_helper.py`:

```python
from tcr_io._resources import resource_path


def test_resource_path_points_to_existing_parquet():
    p = resource_path("hla_model_weights.parquet")
    assert p.exists()
    assert p.name == "hla_model_weights.parquet"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resources_helper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tcr_io._resources'`

- [ ] **Step 3: Create the helper**

Create `tcr_io/_resources.py`:

```python
"""Access to bundled read-only data files under tcr_io/resources/."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def resource_path(name: str) -> Path:
    """Absolute filesystem path to a bundled resource, e.g. 'hla_tcrs.parquet'."""
    return Path(str(files("tcr_io").joinpath("resources", name)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resources_helper.py -v`
Expected: PASS

- [ ] **Step 5: Commit the helper, the generated resources, and the tooling**

```bash
git add tcr_io/_resources.py tests/test_resources_helper.py \
        tcr_io/resources/hla_tcrs.parquet \
        tcr_io/resources/hla_model_weights.parquet \
        tcr_io/resources/cmv_ecocluster.parquet \
        scripts/build_resources.py model_sources/README.md .gitignore
git commit -m "feat: bundle HLA+CMV model parquets and add resource accessor"
```

---

### Task 3: pyproject.toml — dependencies, metadata, maturin include

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: a `pip install`-able project that pulls all runtime deps and bundles `tcr_io/resources/*.parquet` into the wheel.

- [ ] **Step 1: Replace `pyproject.toml` with the fully-specified version**

Overwrite `pyproject.toml` with:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0", "polars>=1.3.0"]
build-backend = "maturin"

[project]
name = "tcrio"
requires-python = ">=3.10"
description = "Polars-based I/O and QC toolkit for T-cell receptor (TCR) repertoire datasets."
readme = "README.md"
keywords = ["TCR", "immunology", "bioinformatics", "polars", "repertoire"]
classifiers = [
  "Programming Language :: Rust",
  "Programming Language :: Python :: Implementation :: CPython",
  "Programming Language :: Python :: Implementation :: PyPy",
]
dynamic = ["version"]
dependencies = [
  # polars is the compiled-plugin ABI boundary — a runtime polars newer than the
  # build can fail to load. <2.0 = loose cap; tightest-safe = <1.43 (built minor).
  "polars>=1.42,<2.0",
  "numpy>=2.2",
  "scipy>=1.14",
  "tqdm>=4.67",
  "packaging>=24",
  "natsort>=8.4",
  "requests>=2.32",
  "matplotlib>=3.8",
  "seaborn>=0.13",
  "networkx>=3.0",
  "plotly>=5.0",
  "colorcet>=3.0",
]

[project.optional-dependencies]
dev = ["maturin>=1.14", "ruff>=0.15", "pytest>=9.1", "mypy>=2.1"]

[project.urls]
Repository = "https://github.com/vincentvandeuren/tcr_qc"

[tool.maturin]
module-name = "tcr_io._internal"
include = ["tcr_io/resources/*.parquet"]

[[tool.mypy.overrides]]
module = "polars.utils.udfs"
ignore_missing_imports = true
```

Note: `license` and `authors` are intentionally omitted for the beta (GitHub Release wheels don't require them); add them before public PyPI. `scikit-learn` is NOT a dependency — the parquet resources already contain the extracted coefficients, so nothing unpickles sklearn models at runtime.

- [ ] **Step 2: Rebuild so the include takes effect and verify the wheel bundles the parquets**

Run:
```bash
maturin build --out dist
python - <<'PY'
import zipfile, glob
whl = sorted(glob.glob("dist/*.whl"))[-1]
names = zipfile.ZipFile(whl).namelist()
res = [n for n in names if n.startswith("tcr_io/resources/") and n.endswith(".parquet")]
print("wheel:", whl)
print("bundled resources:", res)
assert len(res) == 3, res
print("OK")
PY
```
Expected: prints the wheel name, the three `tcr_io/resources/*.parquet` entries, and `OK`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: declare runtime deps, metadata, and bundle resources in wheel"
```

---

### Task 4: HLA operations read bundled parquets

**Files:**
- Modify: `tcr_io/operations/hla.py`
- Test: `tests/test_resources.py`

**Interfaces:**
- Consumes: `tcr_io._resources.resource_path` (Task 2).
- Produces: `HlaInference()` / `RepertoireHlaInference()` constructable with no args. After construction, `self.tcrs` is a `pl.LazyFrame` with columns `{junction_aa, v_gene, j_gene, allele, gene, score}` and `self.wts` a `pl.LazyFrame` with `{allele, coef_1, coef_2, intercept}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_resources.py`:

```python
import pytest


def test_hla_inference_loads_bundled_model():
    from tcr_io.operations import HlaInference

    op = HlaInference()  # no args -> bundled parquets
    tcr_cols = set(op.tcrs.collect_schema().names())
    wt_cols = set(op.wts.collect_schema().names())
    assert {"junction_aa", "v_gene", "j_gene", "allele", "gene", "score"} <= tcr_cols
    assert {"allele", "coef_1", "coef_2", "intercept"} <= wt_cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resources.py::test_hla_inference_loads_bundled_model -v`
Expected: FAIL — the current `__init__` tries to open `/data/current/...` and raises `FileNotFoundError`.

- [ ] **Step 3: Rewrite the loader in `tcr_io/operations/hla.py`**

Replace the import block near the top — delete these lines:

```python
import gzip
import pickle
```

and delete this import:

```python
from ..expressions import to_imgt
```

Add this import (with the other `from ..` imports):

```python
from .._resources import resource_path
```

Then replace the whole `BaseHlaInferenceOperation` `__init__` + `_prepare_tcr_df` + `_prepare_model_wts` methods:

```python
    def __init__(self, model_checkpoint: str | Path | None = None):
        if model_checkpoint is None:
            tcrs_path = resource_path("hla_tcrs.parquet")
            wts_path = resource_path("hla_model_weights.parquet")
        else:
            d = Path(model_checkpoint)
            tcrs_path = d / "hla_tcrs.parquet"
            wts_path = d / "hla_model_weights.parquet"
        self.tcrs = pl.read_parquet(tcrs_path).lazy()
        self.wts = pl.read_parquet(wts_path).lazy()
```

(The parquets are pre-processed, so no gene mapping happens at load. Everything downstream in `_run` — `self.tcrs.with_columns(pl.col("score").abs())`, the join on `["v_gene", "j_gene", "junction_aa"]`, and the `self.wts` join — is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_hla_inference_loads_bundled_model -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tcr_io/operations/hla.py tests/test_resources.py
git commit -m "feat: HLA operations load bundled parquet resources"
```

---

### Task 5: CMV operation reads bundled parquet

**Files:**
- Modify: `tcr_io/operations/cmv_hits.py`
- Test: `tests/test_resources.py`

**Interfaces:**
- Consumes: `tcr_io._resources.resource_path` (Task 2).
- Produces: `ECOClusterHits()` constructable with no args; `self.eco_df` is a `pl.LazyFrame` with columns `{v_gene, j_gene, junction_aa, hla_cocluster, eco_id}`.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_resources.py`:

```python
def test_cmv_hits_loads_bundled_model():
    from tcr_io.operations import ECOClusterHits

    op = ECOClusterHits()  # no args -> bundled parquet
    cols = set(op.eco_df.collect_schema().names())
    assert {"v_gene", "j_gene", "junction_aa", "hla_cocluster", "eco_id"} <= cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resources.py::test_cmv_hits_loads_bundled_model -v`
Expected: FAIL — current `__init__` reads the `/data/current/...CMV_ECOcluster_TCRs.tsv` path.

- [ ] **Step 3: Rewrite `tcr_io/operations/cmv_hits.py` loader**

Delete this import:

```python
from tcr_io.expressions import to_imgt
```

Add:

```python
from tcr_io._resources import resource_path
```

Replace `__init__` and `_prepare_eco_df` with:

```python
    def __init__(self, model_checkpoint: str | Path | None = None):
        if model_checkpoint is None:
            path = resource_path("cmv_ecocluster.parquet")
        else:
            path = Path(model_checkpoint)
        self.eco_df = pl.read_parquet(path).lazy()
```

(Delete the old `_prepare_eco_df` method entirely; the parquet already has the final columns.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_cmv_hits_loads_bundled_model -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tcr_io/operations/cmv_hits.py tests/test_resources.py
git commit -m "feat: CMV operation loads bundled parquet resource"
```

---

### Task 6: MAIT operation requires an explicit checkpoint (deferred)

**Files:**
- Modify: `tcr_io/operations/unconventional_tcrs.py`
- Test: `tests/test_resources.py`

**Interfaces:**
- Produces: `MaitHits()` with no `model_checkpoint` raises `ValueError`; with a path it behaves as before.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_resources.py`:

```python
def test_mait_hits_requires_checkpoint():
    from tcr_io.operations import MaitHits

    with pytest.raises(ValueError, match="requires a model file"):
        MaitHits()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resources.py::test_mait_hits_requires_checkpoint -v`
Expected: FAIL — current `MaitHits()` falls back to the `/data/current/...mait_parsed.parquet` default and raises `FileNotFoundError` (or `ValueError` for the wrong reason).

- [ ] **Step 3: Update `BaseHits.__init__` and drop the hardcoded default**

In `tcr_io/operations/unconventional_tcrs.py`, replace `BaseHits.__init__`:

```python
    def __init__(self, model_checkpoint: Optional[str | Path] = None):
        if model_checkpoint is None:
            raise ValueError(
                f"{type(self).__name__} requires a model file (not bundled in the beta). "
                "Pass model_checkpoint=<path-to-parquet>. See README 'Beta limitations'."
            )
        self.query_df = self._prepare_query_df(Path(model_checkpoint))
```

Delete the `default_model_checkpoint` abstract property from `BaseHits`:

```python
    @property
    @abstractmethod
    def default_model_checkpoint(self) -> Path:
        pass
```

And delete the `default_model_checkpoint` override in `MaitHits`:

```python
    @property
    def default_model_checkpoint(self) -> Path:
        return Path("/data/current/datasets/databases/unconventional_tcrs/data/mait_parsed.parquet")
```

(Leave `MaitHits._prepare_query_df` unchanged — it still reads whatever parquet path is passed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resources.py::test_mait_hits_requires_checkpoint -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tcr_io/operations/unconventional_tcrs.py tests/test_resources.py
git commit -m "feat: MaitHits requires explicit model_checkpoint (deferred from beta)"
```

---

### Task 7: Remove the leaked API key

**Files:**
- Modify: `tcr_io/operations/metadata.py:12-33`

**Interfaces:** none (dead-comment removal).

- [ ] **Step 1: Delete the dead commented block containing the key**

In `tcr_io/operations/metadata.py`, delete the entire commented-out block that starts at `# import pyalex :  can also do abstract` and runs through the end of the commented `get_works` function (the block that contains `"api_key": "2Jm8p6OO4MvFNn0GMqb2b9"`). Remove every line of that comment block.

- [ ] **Step 2: Verify the key is gone**

Run: `grep -rn "2Jm8p6OO4MvFNn0GMqb2b9\|api_key" tcr_io/`
Expected: no output (exit 1).

- [ ] **Step 3: Verify the module still imports**

Run: `python -c "import tcr_io.operations.metadata; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add tcr_io/operations/metadata.py
git commit -m "chore: remove dead comment block with leaked API key"
```

Note for the user (out of scope here): rotate the OpenAlex key and scrub it from git history separately — this only removes it from the current tree.

---

### Task 8: Public API in `tcr_io/__init__.py`

**Files:**
- Modify: `tcr_io/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: `tcr_io.__version__` (str) and top-level names `TcrDataset`, `DatasetIngester`, `ReaderFactory`, `BaseReader`, mappers, `Filterer`, and the `operations` subpackage.

- [ ] **Step 1: Write the failing test**

Create `tests/test_smoke.py`:

```python
def test_import_and_version():
    import tcr_io

    assert isinstance(tcr_io.__version__, str) and tcr_io.__version__


def test_public_api_surface():
    import tcr_io

    for name in ["TcrDataset", "DatasetIngester", "ReaderFactory", "Filterer", "operations"]:
        assert hasattr(tcr_io, name), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL — `tcr_io.__version__` / attributes missing (`__init__.py` is empty).

- [ ] **Step 3: Populate `tcr_io/__init__.py`**

Overwrite `tcr_io/__init__.py` with:

```python
from tcr_io._internal import __version__ as __version__

from .dataset import TcrDataset
from .ingestion import DatasetIngester
from .readers import ReaderFactory, BaseReader
from .mappers import (
    BaseMapper,
    FileNameMapper,
    RegexMapper,
    DictMapper,
    ChainedMapper,
)
from .filters import Filterer
from . import operations

__all__ = [
    "__version__",
    "TcrDataset",
    "DatasetIngester",
    "ReaderFactory",
    "BaseReader",
    "BaseMapper",
    "FileNameMapper",
    "RegexMapper",
    "DictMapper",
    "ChainedMapper",
    "Filterer",
    "operations",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add tcr_io/__init__.py tests/test_smoke.py
git commit -m "feat: expose public API from tcr_io package root"
```

---

### Task 9: Remove the broken tutorial test

**Files:**
- Delete: `tests/test_pig_latinnify.py`

**Interfaces:** none.

- [ ] **Step 1: Confirm it is broken**

Run: `pytest tests/test_pig_latinnify.py -v`
Expected: ERROR on import — `ImportError: cannot import name 'pig_latinnify' from 'tcrio'`.

- [ ] **Step 2: Delete it**

Run: `git rm tests/test_pig_latinnify.py`

- [ ] **Step 3: Run the whole suite green**

Run: `pytest tests -v`
Expected: PASS — `test_smoke.py`, `test_resources.py`, `test_resources_helper.py` all pass; no collection errors.

- [ ] **Step 4: Commit**

```bash
git commit -m "test: remove broken pig_latinnify tutorial test"
```

---

### Task 10: README

**Files:**
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Replace `README.md`**

Overwrite `README.md` with:

```markdown
# tcrio

Polars-based I/O and QC toolkit for T-cell receptor (TCR) repertoire datasets.
Fast ingestion of many sequencing formats into a standardized dataset layout,
plus QC operations (filtering, diversity, gene-usage, VDJ statistics, overlap,
HLA inference, CMV hits).

## Install (closed beta)

Beta wheels are attached to GitHub Releases (not on PyPI yet):

```bash
pip install <url-to-the-release-wheel>
```

The wheel bundles the compiled Rust extension and the HLA/CMV model data, so no
build tools or extra downloads are needed.

## Quick start

```python
import tcr_io

# Build a dataset from a directory of repertoire files
ingester = tcr_io.DatasetIngester(
    db_dir="./my_datasets",
    db_name="example",
    repertoire_mapper=tcr_io.FileNameMapper(),
    patient_mapper=tcr_io.FileNameMapper(),
)
ds = ingester.run("path/to/repertoire/files/*.tsv")

# Run a QC operation
from tcr_io.operations import GeneCountsSummary
ds.run_operation(GeneCountsSummary())
print(ds.get_operation_result("gene_counts_summary"))
```

## Dataset layout

See the module docstring in `tcr_io/dataset.py` for the full on-disk structure
(`processed_repertoires/`, `meta/`, `qc/`, `tabulated/`).

## Beta limitations

- HLA inference (`HlaInference`, `RepertoireHlaInference`) and CMV hits
  (`ECOClusterHits`) work out of the box — their model data is bundled.
- `MaitHits` is **not** bundled in the beta; pass your own parquet:
  `MaitHits(model_checkpoint="path/to/mait.parquet")`.

## Development

```bash
make venv          # create .venv from requirements.txt (dev lockfile)
make install       # maturin develop (build the Rust extension)
make test          # pytest
make pre-commit    # fmt + clippy + ruff + mypy
```

Bundled resources are generated from raw model sources with
`python scripts/build_resources.py --build` (see `model_sources/README.md`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: real README with install, quickstart, and beta limitations"
```

---

### Task 11: Version bump + full verification in a clean venv

**Files:**
- Modify: `Cargo.toml` (version)

**Interfaces:** none — final acceptance gate.

- [ ] **Step 1: Set the beta pre-release version**

In `Cargo.toml`, change:

```toml
version = "0.1.0"
```

to:

```toml
version = "0.1.0-beta.1"
```

- [ ] **Step 2: Build the wheel**

Run: `maturin build --release --out dist`
Expected: exit 0; a `tcrio-0.1.0b1-*.whl` appears in `dist/`.

- [ ] **Step 3: Install into a throwaway clean venv (no system site-packages)**

Run:
```bash
python3 -m venv /tmp/tcrio-clean
/tmp/tcrio-clean/bin/pip install --upgrade pip
/tmp/tcrio-clean/bin/pip install "$(ls -t dist/tcrio-*.whl | head -1)"
```
Expected: pip resolves and installs polars/numpy/scipy/tqdm/packaging/natsort/requests/matplotlib/seaborn/networkx/plotly/colorcet plus `tcrio`, exit 0.

- [ ] **Step 4: Smoke-test the installed package (bundled resources included)**

Run:
```bash
/tmp/tcrio-clean/bin/python - <<'PY'
import tcr_io
print("version:", tcr_io.__version__)
from tcr_io.operations import HlaInference, ECOClusterHits, MaitHits
HlaInference(); ECOClusterHits()          # construct from bundled parquets
try:
    MaitHits(); raise SystemExit("MAIT should have raised")
except ValueError:
    pass
print("clean-venv smoke OK")
PY
```
Expected: prints `version: 0.1.0-beta.1` (the raw Cargo string via `CARGO_PKG_VERSION`; the installed wheel/`pip show` version is the PEP 440 form `0.1.0b1`) and `clean-venv smoke OK`.

- [ ] **Step 5: Run the dev checks (CI proxy)**

Run: `make pre-commit && make test`
Expected: fmt/clippy/ruff/mypy clean, all pytest tests pass.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml
git commit -m "build: set beta pre-release version 0.1.0-beta.1"
```

- [ ] **Step 7: Clean up the throwaway venv**

Run: `rm -rf /tmp/tcrio-clean dist`

---

## Notes for the executor

- Tasks 4–8 import `tcr_io.*`, which requires the compiled extension — run Task 1 (`maturin develop`) first in your venv, and ensure dev deps are installed (`pip install -e ".[dev]"` or `make venv`).
- If `mypy` in Task 11 flags pre-existing issues unrelated to these changes, note them but do not expand scope — this plan targets packaging, not a type-cleanup pass.
- The GitHub-Release-wheel CI wiring (attaching `dist/*.whl` to a release on a `v0.1.0b1` tag) is described in spec §8; implement it only if the user asks — the local build in Task 11 already proves the artifact.
