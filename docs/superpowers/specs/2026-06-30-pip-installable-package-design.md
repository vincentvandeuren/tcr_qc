# Design: Make `tcrio` pip-installable (closed beta → public PyPI)

**Date:** 2026-06-30
**Status:** Approved (design)
**Scope:** Packaging only — make the existing maturin/PyO3 project cleanly `pip install`-able, distributable to a closed beta group, on the same artifacts/flow that public PyPI will later use.

---

## 1. Goal & scope

**Goal:** A clean `pip install` produces a working, importable `tcr_io` package with all runtime dependencies present. It is distributed first to a closed (private) beta group via GitHub Release wheels, then later to public PyPI using the same build pipeline.

### In scope
- Declare runtime dependencies in `pyproject.toml` (currently none are declared).
- Populate the public API (`tcr_io/__init__.py` is currently empty).
- Bundle the **HLA** and **CMV** model files into the package as parquet resources, so those operations work out of the box (see §6).
- The **minimum** set of fixes needed so the build + release pipeline is green and the installed package actually works.
- A real README for beta users.
- A closed-beta distribution mechanism (GitHub Release wheels).

### Explicitly deferred (NOT this effort)
These come from the earlier code-quality report and are intentionally out of scope:
- Dedup/refactors (repeated `v_gene/j_gene` split, duplicated HLA `_run` bodies, etc.).
- Splitting dependencies into `[viz]` / other granular extras (everything is core for now).
- Latent logic bugs (`BaseReader._run_multiple` `None` branch, `is_valid_junction_aa` regex alternation).
- Bundling **MAIT** model data (`MaitHits` is not needed for the beta — keeps the `None` + clear-error treatment).
- Download-on-first-use of model data (we bundle directly instead).
- Broader test coverage of Rust expressions / diversity metrics.

A follow-up effort can pick these up; this spec stays focused on shippability.

---

## 2. Decisions (locked)

| Topic | Decision |
|---|---|
| Final distribution target | Public PyPI |
| First step | Closed (private) beta |
| Beta channel | **GitHub Release wheels** (pre-release tag, e.g. `v0.1.0b1`) — truly private, reuses existing CI |
| PyPI / import name | `tcrio` (distribution) / `tcr_io` (import) — minimum change, can revisit later |
| Dependency layout | **Everything is a core dependency** for now; extras can be split out later |
| Dependency version style | **`>=` lower bounds, no upper caps** in `pyproject.toml`; `requirements.txt` stays exact-pinned (`==`) as the dev/CI lockfile |
| Dev tooling | Moved into a `[project.optional-dependencies].dev` extra |
| HLA + CMV ops | **Bundle model files as parquet resources** in `tcr_io/resources/`; these ops work out of the box from a clean install |
| MAIT op (`MaitHits`) | **Deferred** — not needed for beta; default `model_checkpoint=None` + clear error if used without a path |
| Version source | Continues to come from `Cargo.toml` via `dynamic = ["version"]` |

---

## 3. `pyproject.toml` changes

This is the central fix. Today `[project]` declares **no** `dependencies`, so `pip install tcrio` installs the compiled extension but none of `polars`/`numpy`/`scipy`/etc., and the package is unusable.

### 3.1 Runtime dependencies (all core, `>=` lower bounds)
Add `[project].dependencies`:
```
polars>=1.42
numpy>=2.2
scipy>=1.14
tqdm>=4.67
packaging>=24
natsort>=8.4
requests>=2.32
matplotlib>=3.8
seaborn>=0.13
networkx>=3.0
plotly>=5.0
colorcet>=3.0
```
- Lower bounds derived from the current `requirements.txt` pins (and observed imports). No upper caps (per decision).
- `matplotlib`/`seaborn`/`networkx`/`plotly`/`colorcet` are included even though imported lazily inside `overlap.py` plotting functions — per the "everything core" decision.
- `requests` is required by `operations/metadata.py`.

> **Note on `polars` ABI:** the Rust plugin links a specific `pyo3-polars`/`polars` version. That is a *build-time* (Cargo) constraint, not a Python runtime cap, so it does not add an upper bound to the Python dependency. Beta verification (§9) confirms the built wheel imports and runs against the resolved polars.

### 3.2 Dev extra
Add `[project.optional-dependencies]`:
```
dev = ["maturin>=1.14", "ruff>=0.15", "pytest>=9.1", "mypy>=2.1"]
```
Removes build/test tooling from the runtime set. The Makefile/CI continue to use `requirements.txt` (the exact-pinned dev lockfile) so dev/CI builds stay reproducible.

### 3.3 Public metadata (for PyPI; harmless for beta)
Add to `[project]`: `description`, `readme = "README.md"`, `license`, `authors`, `keywords`, and `[project.urls]` (repository / homepage). Keep `requires-python` consistent with the Rust `abi3-py39` build and the CI test matrix (3.9+); update the `>=3.8` claim to `>=3.9`.

### 3.4 Version
Keep `dynamic = ["version"]`. Version resolves from `Cargo.toml` `package.version`. For the beta, set a **pre-release** version (e.g. `0.1.0b1`) in `Cargo.toml` so the GitHub Release wheel is clearly a beta. Public release later uses a normal version (`0.1.0`).

### 3.5 `requirements.txt`
Stays as the dev/CI lockfile with exact `==` pins. Remove redundancy as needed so it represents the dev environment (runtime deps + dev tooling), but it is **not** the source of truth for what `pip install tcrio` pulls — `pyproject.toml` is.

---

## 4. Public API — `tcr_io/__init__.py`

Currently empty (0 bytes), so a fresh install has nothing meaningful to import. Populate with the intended public surface plus `__version__`:

```python
from tcr_io._internal import __version__ as __version__
from .dataset import TcrDataset
from .ingestion import DatasetIngester
from .readers import ReaderFactory, BaseReader
from .mappers import FileNameMapper, RegexMapper, DictMapper, ChainedMapper, BaseMapper
from .filters import Filterer
from . import operations
```
Define `__all__` accordingly. Goal: `import tcr_io; ds = tcr_io.TcrDataset(path)` works after a clean install. Exact export list to be finalized during implementation by following actual usage, but must at minimum expose `TcrDataset`, `DatasetIngester`, the readers/mappers, and the `operations` subpackage.

---

## 5. Minimum fixes to keep the release pipeline green

The CI (`make pre-commit → install → test`, with `RUSTFLAGS=-Dwarnings`) is currently **red**, which would block any tagged release. Fix exactly these:

1. **Broken test.** Delete `tests/test_pig_latinnify.py` (it imports a `pig_latinnify` symbol that exists nowhere — leftover from the Polars plugin tutorial). Replace with one trivial real smoke test:
   ```python
   def test_import_and_version():
       import tcr_io
       assert isinstance(tcr_io.__version__, str)
   ```
   so `make test` (`pytest tests`) passes.

2. **Rust warnings that fail `-Dwarnings`.**
   - Remove the unused `use rust_stemmers::{Algorithm, Stemmer};` import in `src/expressions.rs`.
   - Remove the now-unused `rust-stemmers` dependency from `Cargo.toml`.
   - Remove the redundant `use std::usize;` in `src/reference_points.rs`.
   - Build/clippy must be clean under `RUSTFLAGS=-Dwarnings` (verify with `make pre-commit`).

3. **Leaked secret.** Remove the OpenAlex `api_key` string in the commented block of `operations/metadata.py`. (User rotates the key on their side; this only removes it from the working tree — history scrubbing is out of scope for this effort but noted.)

---

## 6. Model data — bundle HLA + CMV as parquet resources

Replace the hardcoded `/data/current/...` defaults. Ship the HLA and CMV model files **inside the package** as parquet; leave MAIT deferred.

### 6.1 Resource folder & packaging

- New folder: `tcr_io/resources/` holding the bundled parquet files (committed to the repo; they are small/"not heavy").
- Ensure maturin includes them in the built wheel via `[tool.maturin]`:
  ```
  include = ["tcr_io/resources/*.parquet"]
  ```
  (Verify the built wheel actually contains them — see §9.)
- Access at runtime with `importlib.resources`, not a relative `__file__` path:
  ```python
  from importlib.resources import files
  RESOURCES = files("tcr_io.resources")
  ```

### 6.2 HLA (`HlaInference`, `RepertoireHlaInference`)

The model comes from a published paper; the user generates two **pre-processed** parquet files (the Rust `to_imgt`/gene-split is baked in at generation time):

- `hla_tcrs.parquet` — columns: `junction_aa`, `v_gene`, `j_gene`, `allele`, `gene`, `score`
- `hla_model_weights.parquet` — columns: `allele`, `coef_1`, `coef_2`, `intercept`

Code changes in `operations/hla.py`:
- `BaseHlaInferenceOperation.__init__(model_checkpoint=None)`. When `None`, load the two bundled parquets from `RESOURCES`. When a path is given, treat it as a directory containing those same two parquet filenames (override hook for users with their own models).
- `_prepare_tcr_df` → `pl.read_parquet(<dir>/"hla_tcrs.parquet").lazy()` (one line; the schema above is already the final consumed schema — joins use `["v_gene","j_gene","junction_aa"]`).
- `_prepare_model_wts` → `pl.read_parquet(<dir>/"hla_model_weights.parquet").lazy()`.
- **Remove** the `gzip` + `pickle` loading entirely (eliminates the pickle security/portability risk for public PyPI).

### 6.3 CMV (`ECOClusterHits`)

- Convert the source TSV (`CMV_ECOcluster_TCRs.tsv`) to a single bundled parquet, e.g. `cmv_ecocluster.parquet`. The user generates it; the existing `_prepare_eco_df` parsing/`to_imgt` may be baked in so the parquet carries the final consumed columns (`v_gene`, `j_gene`, `junction_aa`, `hla_cocluster`, `eco_id`), OR the parquet mirrors the TSV columns and `_prepare_eco_df` keeps its transform reading from parquet. Implementation picks whichever the generated file matches; default is to bake in (consistent with HLA).
- `ECOClusterHits.__init__(model_checkpoint=None)` → when `None`, load the bundled `cmv_ecocluster.parquet` from `RESOURCES`.

### 6.4 MAIT (`MaitHits`) — deferred

- `MaitHits.__init__(model_checkpoint=None)`. When `None`, raise a clear error:
  ```
  ValueError("MaitHits requires a model file (not bundled in the beta). "
             "Pass model_checkpoint=<path-to-parquet>. See README 'Beta limitations'.")
  ```
- No bundled resource; no behavior change when a path *is* provided.

### 6.5 Prerequisite / handoff

Generating `hla_tcrs.parquet`, `hla_model_weights.parquet`, and `cmv_ecocluster.parquet` (with the schemas above) is a **user-provided input**. Implementation can scaffold the code, defaults, and packaging against these filenames/schemas, but the bundled ops cannot be fully verified end-to-end until the real files are dropped into `tcr_io/resources/`. Tracked as a risk in §10.

---

## 7. README

Replace the one-line `# tcrust` with a real beta README containing:
- What `tcrio` is (one paragraph).
- Install instructions for the beta (GitHub Release wheel / `pip install <wheel-url>` — see §8).
- A minimal end-to-end example: construct a `DatasetIngester`, ingest a small input, run one file-free QC operation, read a result.
- The dataset directory-layout reference (surface the existing excellent docstring from `dataset.py`).
- A **"Beta limitations"** section: HLA and CMV operations work out of the box (models bundled); `MaitHits` is not bundled in the beta and requires a user-supplied parquet via `model_checkpoint=`.

---

## 8. Beta distribution mechanism (GitHub Release wheels)

- Tag a pre-release (e.g. `v0.1.0b1`).
- Existing `.github/workflows/publish_to_pypi.yml` already builds wheels for Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows, and an sdist. The `release` job only uploads to PyPI on tags **and** is gated on the `pypi` environment/token — for the beta we do **not** publish to PyPI; instead we attach the built wheels to a **GitHub Release**.
- Implementation detail: add/adjust a workflow path so that on a pre-release tag, the built `wheels-*` artifacts are published to a GitHub Release (e.g. via `softprops/action-gh-release` or `gh release create`) rather than (or in addition to skipping) the PyPI upload. The PyPI upload path remains for the eventual public, non-prerelease tag.
- Beta testers install with `pip install <url-to-wheel>` (or from a private repo checkout). Nothing is published to any public index.
- **Public release later** = the same pipeline with a normal version tag (`v0.1.0`) + the existing PyPI token path; no rework.

---

## 9. Verification

Before declaring done:
1. In a **clean virtualenv**, `pip install` the built wheel (not an editable/dev install) → `python -c "import tcr_io; print(tcr_io.__version__)"` succeeds.
2. Run a minimal end-to-end flow: ingest a tiny sample input via `DatasetIngester` + a reader, run one file-free operation (e.g. `DiversityReport` or `GeneCountsSummary`), read back a result — no missing-dependency import errors.
3. **Bundled resources load:** confirm the built wheel contains `tcr_io/resources/*.parquet`, and that `HlaInference()` / `ECOClusterHits()` construct with **no** arguments (loading bundled parquets) and run against a tiny repertoire — out of the box, no path needed.
4. `MaitHits()` with no `model_checkpoint` → confirms the clear `ValueError` (not a `/data/current` `FileNotFoundError`).
5. `make pre-commit && make test` is green locally (proxy for CI).
6. (If CI workflow touched) Confirm a pre-release tag produces a GitHub Release with attached wheels.

---

## 10. Risks / notes

- **Bundled model files are a user-provided prerequisite.** The HLA (`hla_tcrs.parquet`, `hla_model_weights.parquet`) and CMV (`cmv_ecocluster.parquet`) files must be generated by the user and placed in `tcr_io/resources/` before the bundled ops can be verified end-to-end (§6.5). Code, defaults, and packaging can be built against the agreed filenames/schemas in parallel, but final §9.3 verification is blocked on these files.
- **History scrubbing of the leaked key** is out of scope (removing from the working tree ≠ removing from git history). Flagged for the user to handle separately (rotate the key regardless).
- **`requires-python` bump** from `>=3.8` to `>=3.9` is a (minor) compatibility narrowing, justified by the existing `abi3-py39` Rust build and CI matrix already excluding 3.8.
- **No upper caps** means a future breaking dependency major (e.g. `polars 2.0`) could install and break at runtime; acceptable for a beta, revisit for public GA.
