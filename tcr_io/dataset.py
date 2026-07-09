"""
Dataset structure

The authoritative, machine-readable definition of this layout lives in `tcr_io/layout.py`
(the `Layout` registry). The tree below is illustrative only — resolve paths via
`Layout.<key>.path`, never by hardcoding these strings.

my_dataset/
├── processed_repertoires/      # One parquet per repertoire. Atomic unit.
│   ├── sample_001.parquet      # Standardized schema, repertoire_id column
│   └── sample_002.parquet
│
├── tabulated/                  # Hive-partitioned. Generated, deletable, rebuildable.
│   ├── v_gene=TRBV7-2/
│   │   └── j_gene=TRBJ2-1/
│   │       └── my_dataset.parquet
│   └── ...
│
├── meta/
│   ├── generation.json # One row per dataset generation. Metadata about when, how, source, etc.
│   ├── operations.json # One row per operation. Metadata about ops performed on this dataset
│   ├── repertoire/
│   │   ├── repertoire.parquet  # One row per repertoire. IDs, counts, source files, patient_ids. 
│   │   └── repertoire_meta.parquet # Optional, additional metadata about repertoires (e.g. source, processing notes). Must have repertoire_id column to join with repertoire.parquet.
│   ├── patient/
│   │   ├── patient.parquet     # One row per patient. Aggregated stats.
│   │   ├── patient_meta.parquet     # Optional, additional metadata about patients (e.g. clinical notes). Must have patient_id column to join with patient.parquet.
│   │   ├── hla.parquet         # Optional. Known HLA typing.
│   │   └── inferred_hla.parquet # Optional. Computationally inferred.
│   └── publication/
│       ├── publication_ids.json # DOI(s), pubmed_id(s) or other identifiers for publications associated with this dataset.
│       ├── pdfs                # optional, generated or manually added publication PDFs. 
│       └── publication.parquet # Generated,fetched metadata cached here
│
├── qc/                         # Generated. QC metric tables + plots.
│   ├── repertoire_stats.parquet
│   ├── gene_usage.parquet
│   ├── overlap.parquet
│   └── repertoire_stats.png
│
└── README.md                   # Optional. Auto-generated dataset card.
"""
from __future__ import annotations
from functools import cached_property
import json
import logging
import warnings
from time import time
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple
import polars as pl
from pathlib import Path
from tqdm import tqdm
from packaging.version import parse as parse_version

from .operations.base import OperationFailure, OperationResults
from .operations import BaseOperation, OperationMeta
from .grouper import Grouper
from .structure import Layout, REQUIRED_DIRS, repertoire_relpath, Manifest, DATASET_VERSION

log = logging.getLogger(__name__)


class TcrDataset:
    """
    Thin read-only handle to a TCR dataset directory.
    Knows structure. Provides accessors. No mutation logic.
    """

    def __init__(self, db_dir:str|Path):
        self.db_dir = Path(db_dir)
        self.db_name = self.db_dir.name
        self._validate_dirs()
        self._check_version()

    @classmethod
    def migrated(cls, db_dir:str|Path) -> "TcrDataset":
        """Alternative constructor: open a dataset, upgrading it **in place** to the current
        version first. Unlike ``TcrDataset(path)`` (which only warns when behind), this mutates
        the dataset on disk via ``.migrate()`` and returns a handle at ``DATASET_VERSION``."""
        ds = cls.__new__(cls)
        ds.db_dir = Path(db_dir)
        ds.db_name = ds.db_dir.name
        if not ds.db_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {ds.db_dir}")
        ds.migrate()          # brings structure AND version up to current, on a possibly-old tree
        ds._validate_dirs()   # now the tree matches the current layout
        ds._check_version()   # equal -> silent
        return ds

    def path(self, artifact) -> Path:
        """Absolute path of a layout Artifact within this dataset."""
        return self.db_dir / artifact.path

    @cached_property
    def _manifest(self) -> Manifest:
        return Manifest.read(self.db_dir / Layout.manifest.path)

    @property
    def version(self) -> int:
        return self._manifest.version

    def _validate_dirs(self):
        if not self.db_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.db_dir}")

        missing = [d for d in REQUIRED_DIRS if not (self.db_dir / d).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required subdirectories: {', '.join(missing)}")

    def _check_version(self):
        disk, lib = self.version, DATASET_VERSION
        match (disk > lib) - (disk < lib):        # -1 behind, 0 equal, +1 ahead
            case 0:
                return
            case -1:
                warnings.warn(
                    f"Dataset is v{disk}, library expects v{lib}. "
                    f"Open with TcrDataset.migrated(path) to upgrade it in place.",
                    stacklevel=2,
                )
            case 1:
                raise RuntimeError(
                    f"Dataset is v{disk} but this tcrio only understands v{lib}. Upgrade tcrio."
                )

    def migrate(self, target: int = DATASET_VERSION, *, dry_run: bool = False) -> list:
        """Apply registered migrations to bring the dataset up to ``target``. Commits the
        manifest after each step so a mid-chain failure leaves a resumable state."""
        from .structure import migrations   # lazy: avoids dataset<->migrations import cycle

        plan, v = [], self.version
        while v < target:
            step = migrations.REGISTRY.get(v + 1)
            if step is None:
                raise RuntimeError(f"No migration registered for v{v} -> v{v + 1}")
            plan.append(step)
            v = step.to_version

        for step in plan:
            log.info("migrate v%d -> v%d: %s", step.to_version - 1, step.to_version, step.description)
            if not dry_run:
                step.fn(self)
                self._write_version(step.to_version)   # commit after each step
        return plan

    def _write_version(self, v: int) -> None:
        Manifest(version=v, tcrio_version=Manifest.current().tcrio_version).write(
            self.db_dir / Layout.manifest.path
        )
        self.__dict__.pop("_manifest", None)   # invalidate cached_property -> version re-reads
            
    def run_operation(self, operation:BaseOperation, force = False):
        if self._operation_exists(operation) and not force:   
            warnings.warn(f"Operation {operation.name} v{operation.version} has already been run successfully on this dataset. Skipping.")
            return
        
        operation_res = operation.run(self)
        if isinstance(operation_res, OperationFailure):
            warnings.warn(f"Operation {operation.name} failed with error: {operation_res.error}", stacklevel=2)

        self._write_operation_results(operation, operation_res)
            
    def _write_operation_results(self, operation:BaseOperation, operation_res:OperationResults | OperationFailure):

        match operation_res:
            case OperationFailure():
                outputs, status, error = [], "failure", operation_res.error
            case OperationResults():
                outputs, status, error = list(operation_res.outputs.keys()), "success", None
        
        if isinstance(operation_res, OperationResults):
            for output_path, result in operation_res.outputs.items():
                out_path = self.db_dir/output_path
                if not out_path.parent.exists():
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                match result:
                    case pl.DataFrame():
                        print(f"Writing DataFrame to {out_path}")
                        result.write_parquet(out_path.with_suffix(".parquet"))
                    case pl.LazyFrame():
                        print(f"Writing LazyFrame to {out_path}")
                        result.sink_parquet(out_path.with_suffix(".parquet"))
                    case list():
                        ndjson_path = out_path.with_suffix(".json")
                        print(f"Writing NDJSON to {ndjson_path}")
                        with open(ndjson_path, "w") as f:
                            for record in result:
                                f.write(json.dumps(record, default=str) + "\n")
                    case dict():
                        raise NotImplementedError("Dict outputs not implemented yet")
                    case _:
                        warnings.warn(f"Output type {type(result)} not recognized. Skipping write for output {output_path}.")
        
        operation_meta_new = pl.DataFrame({
            "operation_name" : operation.name,
            "version" : operation.version,
            "ran_at": datetime.now(),
            "duration_s": operation_res.duration_s,
            "status" : status,
            "error" : error,
            "description" : operation.description,
            "outputs" : [outputs]
        }, schema=Layout.operations_meta.schema)

        pl.concat([self.operations_meta, operation_meta_new]).write_ndjson(
            self.db_dir / Layout.operations_meta.path
        )

    def get_operation_result(self, operation_name, output_name:Optional[str] = None) -> pl.DataFrame:
        """
        if output_name is None, returns the first output of the operation. Otherwise, looks for an output containing output_name in its path.
        """
        for op in self.operations[::-1]: # newest first
            if op.error:
                warnings.warn(f"Operation {op.operation_name} (version {op.version}) failed. Skipping.")
                continue
            if op.operation_name == operation_name:
                for output in op.outputs:
                    if output_name is None or output_name in output:
                        output_path = self.db_dir / output
                        if output_path.exists():
                            if output_path.suffix == ".parquet":
                                return pl.read_parquet(output_path)
                            elif output_path.suffix == ".json":
                                return pl.read_ndjson(output_path)
                        else:
                            warnings.warn(f"Output file {output_path} does not exist. Skipping.")
                possible_outputs = ", ".join(op.outputs)
                raise ValueError(f"No output containing {output_name} found for operation {operation_name}. Valid outputs for this operation were: {possible_outputs}")
        raise ValueError(f"No successful operation named {operation_name} with output containing {output_name} found.")

    def _write_meta_parquet(self, subdir:str, name:str, df : pl.DataFrame | pl.LazyFrame):
        path = self.db_dir / subdir / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(df, pl.LazyFrame):
            df.sink_parquet(path)
        else:
            df.write_parquet(path)

    @property
    def repertoire_dir(self) -> Path:
        return self.db_dir / Layout.processed_dir.path

    @property
    def repertoire_meta(self) -> pl.DataFrame:
        art = Layout.repertoire_meta
        return (
            pl.read_parquet(self.db_dir / art.path)
            .select(art.schema.keys())
            .cast(art.schema)
        )

    @property
    def full_repertoire_meta(self) -> pl.DataFrame:
        rep = self.repertoire_meta

        extra = self.db_dir / Layout.repertoire_meta_extra.path
        if extra.exists():
            rep = rep.join(pl.read_parquet(extra), on="repertoire_id", how="left")

        return rep

    @property
    def patient_meta(self) -> pl.DataFrame:
        art = Layout.patient_meta
        return (
            pl.read_parquet(self.db_dir / art.path)
            .select(art.schema.keys())
            .cast(art.schema)
        )

    @property
    def full_patient_meta(self) -> pl.DataFrame:
        pat = self.patient_meta

        extra = self.db_dir / Layout.patient_meta_extra.path
        if extra.exists():
            pat = pat.join(pl.read_parquet(extra), on="patient_id", how="left")

        hla = self.db_dir / Layout.hla.path
        if hla.exists():
            pat = pat.join(pl.read_parquet(hla), on="patient_id", how="left")

        return pat

    @cached_property
    def publication_meta(self) -> dict:
        art = Layout.publication_ids
        return (
            pl.read_ndjson(self.db_dir / art.path, schema=art.schema)
        ).to_dict(as_series=False)

    @cached_property
    def full_publication_meta(self) -> pl.DataFrame:
        pub = pl.DataFrame(self.publication_meta)

        pub_extra = self.db_dir / Layout.publication_meta.path
        if pub_extra.exists():
            pub = pub.join(pl.read_parquet(pub_extra), on="publication_id", how="left")

        return pub

    @cached_property
    def generation_meta(self) -> dict:
        art = Layout.generation_meta
        return (
            pl.read_ndjson(self.db_dir / art.path, schema=art.schema)
        ).to_dicts()[0]

    @property
    def operations_meta(self) -> pl.DataFrame:
        art = Layout.operations_meta
        return (
            pl.read_ndjson(self.db_dir / art.path, schema=art.schema)
        )

    @property
    def operations(self) -> List[OperationMeta]:
        operations = []
        with open(self.db_dir / Layout.operations_meta.path, "r") as f:
            for line in f.readlines():
                json_obj = json.loads(line)
                operations.append(OperationMeta(**json_obj))
        return operations
    
    def _operation_exists(self, operation:BaseOperation) -> bool:
        for op in self.operations:
            if op.operation_name == operation.name and parse_version(op.version) >= parse_version(operation.version) and op.status == "success":
                return True
        return False
    
    def iter_repertoires(self, lazy=True, progress_bar=False, filter_pass_only=True, progress_desc:Optional[str]=None) -> Generator[str, pl.DataFrame | pl.LazyFrame]:
        if progress_bar:
            desc = progress_desc or "Iterating repertoires"
            progress = tqdm(total=self.n_repertoires, desc=desc)

        for parquet_file in self.repertoire_dir.glob("*.parquet"):
            repertoire_id = pl.scan_parquet(parquet_file).select(pl.col("repertoire_id").first()).collect()[0, 0]
            df = pl.scan_parquet(parquet_file)
            if filter_pass_only:
                df = df.filter(pl.col("filter_pass"))
            if lazy:
                yield repertoire_id, df
            else:
                yield repertoire_id, df.collect(engine="streaming")
            if progress_bar:
                progress.update(1)

    def iter_repertoires_by_patient(
            self,
            deduplicate=True,
            filter_pass_only=True,
            lazy=True,
            progress_bar=False,
            progress_desc:Optional[str]=None
            ) -> Generator[Tuple[str, pl.DataFrame | pl.LazyFrame], None, None]:
        patient_meta = self.full_patient_meta

        if progress_bar:
            desc = progress_desc or "Iterating repertoires by patient"
            progress = tqdm(total=patient_meta.select(pl.col("patient_id").n_unique())[0, 0], desc=desc)

        for patient, patient_repertoires, in self.patient_meta.select("patient_id", "patient_repertoires").iter_rows():
            files = [self.db_dir / repertoire_relpath(rep_id) for rep_id in patient_repertoires]
            n_files = len(files)

            df = pl.concat([pl.scan_parquet(f).with_columns(file=pl.lit(f.name)) for f in files])
            if filter_pass_only:
                df = df.filter(pl.col("filter_pass"))

            if deduplicate:
                if n_files > 1:
                    df  = Grouper().run(df)
            
            if not lazy:
                df = df.collect(engine="streaming")
            
            df = df.with_columns(
                pl.lit(patient).alias("patient_id")
            )
            
            yield patient, df

            if progress_bar:
                progress.update(1)
    
    @cached_property
    def n_repertoires(self):
        return self.repertoire_meta.select(pl.col("repertoire_id").n_unique())[0, 0]
    
    @cached_property
    def n_patients(self):
        return self.patient_meta.select(pl.col("patient_id").n_unique())[0, 0]
    
    @cached_property
    def n_clonotypes(self):
        return self.repertoire_meta.select(pl.sum("n_clonotypes"))[0, 0]
    
    def __repr__(self):
        return f"TcrDataset \'{self.db_name}\', created_on {self.generation_meta['created_on']}, {self.n_clonotypes} clonotypes ({self.n_repertoires} repertoires, {self.n_patients} patients)."
