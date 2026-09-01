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
print(ds.result(GeneCountsSummary.gene_counts, locus="TRB").read())
```

## Dataset layout

A dataset is a directory with this on-disk structure:

```
my_dataset/
├── processed_repertoires/          # One directory per repertoire, hive-partitioned by locus.
│   └── sample_001/                 # Atomic unit: standardized schema, repertoire_id column.
│       ├── locus=TRB/sample_001.parquet
│       └── locus=TRA/sample_001.parquet
│
├── meta/
│   ├── manifest.json           # Dataset version, present loci, ingest filter provenance.
│   ├── generation.json         # One row per dataset generation. When, how, source, etc.
│   ├── repertoire/
│   │   ├── TRB.parquet              # One row per repertoire on this locus. IDs, counts.
│   │   └── repertoire_meta.parquet  # Optional extra metadata (join on repertoire_id).
│   ├── patient/
│   │   ├── patient.parquet           # One row per patient. Aggregated stats.
│   │   ├── patient_meta.parquet      # Optional extra metadata (join on patient_id).
│   │   └── hla.parquet               # Optional. Known HLA typing.
│   ├── publication/
│   │   ├── publication_ids.json      # DOI(s) / pubmed_id(s) for associated publications.
│   │   └── publication.parquet       # Generated, fetched metadata cached here.
│   └── clone_to_cell/                # Single-cell datasets only: clonotype <-> cell map.
│
└── operations/                 # Generated, deletable, rebuildable.
    └── <operation>/<LOCUS>/    # One pass per (operation, locus); no locus segment when
        ├── operation.json      # the operation is locus-agnostic.
        └── *.parquet           # The pass's declared outputs, read back via `ds.result()`.
```

## Bundled models

All model-based operations work out of the box — their reference data ships
inside the wheel and loads automatically with no arguments:

- `HlaInference` / `RepertoireHlaInference` — HLA inference
- `ECOClusterHits` — CMV EcoCluster hits
- `MaitHits` — MAIT hits

Each also accepts `model_checkpoint=<path>` to override with your own model.

## Development

```bash
make venv          # create .venv from requirements.txt (dev lockfile)
make install       # maturin develop (build the Rust extension)
make test          # pytest
make pre-commit    # fmt + clippy + ruff + mypy
```

Bundled resources are generated from raw model sources with
`python scripts/build_resources.py --build` (see `model_sources/README.md`).
