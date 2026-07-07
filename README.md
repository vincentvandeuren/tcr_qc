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
