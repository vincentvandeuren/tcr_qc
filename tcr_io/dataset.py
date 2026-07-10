"""
Dataset structure

The authoritative, machine-readable definition of this layout lives in `tcr_io/layout.py`
(the `Layout` registry). The tree below is illustrative only — resolve paths via
`Layout.<key>.path`, never by hardcoding these strings.

my_dataset/
├── processed_repertoires/      # One directory per repertoire, hive-partitioned by locus.
│   ├── sample_001/             # Written in one pl.PartitionBy streaming pass.
│   │   ├── locus=TRB/sample_001.parquet   # Standardized schema; locus is in the PATH, not the file.
│   │   ├── locus=TRA/sample_001.parquet
│   │   └── locus=_unassigned/sample_001.parquet   # QC: rows with no derivable locus (excluded from loci).
│   └── sample_002/
│       └── locus=TRB/sample_002.parquet
│
├── meta/                       # Canonical dataset tables ONLY (no op outputs).
│   ├── generation.json # One row per dataset generation. Metadata about when, how, source, etc.
│   ├── repertoire/
│   │   ├── TRB.parquet         # One row per repertoire in this locus. IDs, counts, source files, patient_ids.
│   │   ├── TRA.parquet         # One meta table per present locus (Option C).
│   │   └── repertoire_meta.parquet # Optional, additional metadata about repertoires (e.g. source, processing notes). Must have repertoire_id column to join.
│   ├── patient/
│   │   ├── patient.parquet     # One row per patient. Aggregated stats.
│   │   ├── patient_meta.parquet     # Optional, additional metadata about patients (e.g. clinical notes). Must have patient_id column to join with patient.parquet.
│   │   └── hla.parquet         # Optional. Known (real) HLA typing. Inferred HLA is an op result under operations/.
│   └── publication/
│       ├── publication_ids.json # DOI(s), pubmed_id(s) or other identifiers for publications associated with this dataset.
│       ├── pdfs                # optional, generated or manually added publication PDFs.
│       └── publication.parquet # Generated,fetched metadata cached here
│
├── operations/                 # Generated. All operation results, one dir per op.
│   └── <op_name>/
│       ├── operation.json      # Self-describing per-op record (version, params, outputs, loci, ...).
│       └── [<LOCUS>/]<output>.<ext>   # Named outputs; locus segment for locus-aware ops.
│
└── README.md                   # Optional. Auto-generated dataset card.
"""
from __future__ import annotations
from functools import cached_property
from dataclasses import asdict
import json
import logging
import shutil
import warnings
from time import time
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple
import polars as pl
from pathlib import Path
from tqdm import tqdm
from packaging.version import parse as parse_version

from .operations.base import (
    OperationFailure, OperationResults, OperationRecord, OutputRecord,
)
from .operations import BaseOperation
from .grouper import Grouper
from .structure import (
    Layout, REQUIRED_DIRS, repertoire_dir_relpath, repertoire_locus_relpath, loci_glob,
    repertoire_meta_relpath, operation_relpath,
    write_artifact, Artifact, Kind, Role, Manifest, DATASET_VERSION,
)
from .structure import schema as _schema
from .expressions import UNASSIGNED

log = logging.getLogger(__name__)


# Extension chosen from an output's Kind at write time (the op names outputs; the framework
# roots + suffixes them). Unstructured outputs are op-written directories -> no extension and
# no framework writer, so they never pass through here.
_KIND_EXT = {
    Kind.PARQUET: ".parquet",
    Kind.NDJSON: ".ndjson",
    Kind.JSON: ".json",
}


def _output_kind(result) -> Kind:
    if isinstance(result, list):
        return Kind.NDJSON
    if isinstance(result, (pl.DataFrame, pl.LazyFrame)):
        return Kind.PARQUET
    raise TypeError(f"Unsupported operation output type: {type(result)!r}")


def _read_output(path: Path, kind: str):
    if kind == Kind.PARQUET.value:
        return pl.read_parquet(path)
    if kind == Kind.NDJSON.value:
        return pl.read_ndjson(path)
    if kind == Kind.JSON.value:
        return json.loads(Path(path).read_text())
    raise ValueError(f"Cannot read operation output of kind {kind!r} at {path}")


class TcrDataset:
    """
    Thin read-only handle to a TCR dataset directory.
    Knows structure. Provides accessors. No mutation logic.
    """

    def __init__(self, db_dir:str|Path):
        self.db_dir = Path(db_dir)
        self.db_name = self.db_dir.name
        # Unstructured directory outputs an op requested during the current `_run` (reset per
        # locus in `run_operation`); drained into the op record. See `_operation_output_dir`.
        self._pending_unstructured: List[OutputRecord] = []
        self._validate_dirs()
        self._check_version()
        self._refresh_operation_results()

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
        ds._pending_unstructured = []
        ds.migrate()          # brings structure AND version up to current, on a possibly-old tree
        ds._validate_dirs()   # now the tree matches the current layout
        ds._check_version()   # equal -> silent
        ds._refresh_operation_results()
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
        # Preserve present_loci (and any other manifest fields set by ingestion/migration) when a
        # migration bumps the version — reconstructing a bare Manifest here would wipe them.
        cur = Manifest.read(self.db_dir / Layout.manifest.path)
        Manifest(version=v, tcrio_version=Manifest.current().tcrio_version,
                 present_loci=cur.present_loci).write(self.db_dir / Layout.manifest.path)
        self.__dict__.pop("_manifest", None)   # invalidate cached_property -> version re-reads

    def _present_loci(self) -> frozenset:
        """Loci present in the dataset — recorded in the manifest at ingestion and fixed thereafter
        (O(1) read, no globbing). Drives the per-locus fan-out in `run_operation`. Excludes the
        `_unassigned` QC bucket (never recorded as a present locus)."""
        return frozenset(self._manifest.present_loci)

    def run_operation(self, operation: BaseOperation, *, force: bool = False):
        """Run an operation across the loci it supports, writing name-keyed outputs under
        ``operations/<op>/[<locus>/]`` and a self-describing ``operation.json`` record.

        Per-locus skip: a locus whose successful result is already recorded (at >= this op's
        version) is skipped unless ``force``; only missing loci re-run."""
        present = self._present_loci()
        ran_loci: List[Optional[str]] = []
        output_records: List[OutputRecord] = []
        start = time()

        for locus in operation.loci_to_run(present):
            if not force and self._operation_done(operation, locus):
                continue
            self._pending_unstructured = []          # unstructured dirs the op writes this pass
            res = operation.run(self, locus)
            if isinstance(res, OperationFailure):
                warnings.warn(f"Operation {operation.name} failed with error: {res.error}", stacklevel=2)
                self._write_operation_record(
                    operation, status="failure", error=res.error,
                    duration_s=time() - start, ran_loci=ran_loci, output_records=output_records,
                )
                return
            for name, result in res.outputs.items():
                output_records.append(self._write_output(operation, name, result, locus))
            output_records.extend(self._pending_unstructured)   # op-written directory outputs
            ran_loci.append(locus)

        if not ran_loci:
            warnings.warn(
                f"Operation {operation.name} v{operation.version} is already complete for all "
                f"present loci; nothing to run. Pass force=True (or use ds.rerun) to recompute.",
                stacklevel=2,
            )
            return

        self._write_operation_record(
            operation, status="success", error=None,
            duration_s=time() - start, ran_loci=ran_loci, output_records=output_records,
        )
        self._refresh_operation_results()   # make the just-run op tab-completable

    def _write_output(self, operation: BaseOperation, name: str, result, locus: Optional[str]) -> OutputRecord:
        kind = _output_kind(result)
        art = Artifact(operation_relpath(operation.name, name, locus) + _KIND_EXT[kind], kind, role=Role.GENERATED)
        write_artifact(art, result, self.db_dir)
        op_prefix = f"{Layout.operations_dir.path}/{operation.name}/"
        return OutputRecord(name=name, path=art.path[len(op_prefix):], kind=kind.value, locus=locus)

    def _operation_output_dir(self, operation: BaseOperation, name: str, locus: Optional[str] = None) -> Path:
        """Allocate THE unstructured-output directory for `operation` (one per op, per locus).

        The op writes its files (and any subdirs) here directly, then need not return anything
        for it — the framework records it as a ``Kind.UNSTRUCTURED`` output after ``_run``.
        The dir is emptied + created so a forced re-run never inherits stale files. A second
        call for the same locus raises: an op gets one unstructured dir per locus and should
        create subdirs inside it if it needs more. Only valid while an op is running (it
        appends to the per-pass ``_pending_unstructured`` list drained by `run_operation`)."""
        if any(p.locus == locus for p in self._pending_unstructured):
            raise ValueError(
                f"{operation.name} already has an unstructured output dir for locus={locus!r}; "
                f"an operation gets one per locus — create subdirs inside it instead."
            )
        relpath = operation_relpath(operation.name, name, locus)
        d = self.db_dir / relpath
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        op_prefix = f"{Layout.operations_dir.path}/{operation.name}/"
        self._pending_unstructured.append(
            OutputRecord(name=name, path=relpath[len(op_prefix):], kind=Kind.UNSTRUCTURED.value, locus=locus)
        )
        return d

    def _operation_dir(self, op_name: str) -> Path:
        return self.db_dir / Layout.operations_dir.path / op_name

    def _read_operation_record(self, op_name: str) -> Optional[OperationRecord]:
        path = self._operation_dir(op_name) / "operation.json"
        if not path.exists():
            return None
        return OperationRecord.from_dict(json.loads(path.read_text()))

    def _write_operation_record(self, operation: BaseOperation, *, status: str, error: Optional[str],
                                duration_s: float, ran_loci: List[Optional[str]],
                                output_records: List[OutputRecord]) -> None:
        outputs = list(output_records)
        loci = list(ran_loci)
        # Merge with a prior successful record so a partial (per-locus) re-run accumulates
        # loci/outputs rather than clobbering the ones we did not re-run this time.
        prior = self._read_operation_record(operation.name)
        if status == "success" and prior is not None and prior.status == "success":
            outputs += [o for o in prior.outputs if o.locus not in ran_loci]
            loci += [l for l in (prior.loci or []) if l not in loci]

        record = OperationRecord(
            operation_name=operation.name,
            version=operation.version,
            description=operation.description,
            ran_at=datetime.now(),
            duration_s=duration_s,
            status=status,
            error=error,
            params=operation.params(),
            loci=[l for l in loci if l is not None] or None,
            outputs=outputs,
        )
        path = self._operation_dir(operation.name) / "operation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(record), indent=2, default=str))

    def _operation_done(self, operation: BaseOperation, locus: Optional[str] = None) -> bool:
        rec = self._read_operation_record(operation.name)
        if rec is None or rec.status != "success":
            return False
        if parse_version(rec.version) < parse_version(operation.version):
            return False
        if locus is not None and (rec.loci is None or locus not in rec.loci):
            return False
        return True

    def get_operation_result(self, operation_name: str, output: Optional[str] = None,
                             locus: Optional[str] = None):
        """Read one operation output by exact name (deterministic — no substring/suffix guessing).

        ``output=None`` returns the sole/first output; ``locus`` selects a per-locus copy.
        Raises with the available ``(name, locus)`` pairs on a miss."""
        rec = self._read_operation_record(operation_name)
        if rec is None:
            raise ValueError(f"No operation record found for '{operation_name}'.")
        if rec.status != "success":
            raise ValueError(f"Operation '{operation_name}' did not succeed (status={rec.status}).")

        candidates = rec.outputs
        if locus is not None:
            candidates = [o for o in candidates if o.locus == locus]
        if output is not None:
            candidates = [o for o in candidates if o.name == output]
        if not candidates:
            available = [(o.name, o.locus) for o in rec.outputs]
            raise ValueError(
                f"No output (name={output!r}, locus={locus!r}) for operation '{operation_name}'. "
                f"Available (name, locus): {available}"
            )

        orec = candidates[0]
        path = self._operation_dir(operation_name) / orec.path
        if orec.kind == Kind.UNSTRUCTURED.value:
            return path          # an op-written directory; the caller reads its files itself
        return _read_output(path, orec.kind)

    def _refresh_operation_results(self) -> None:
        """(Re)build the autocompleting ``operation_results`` namespace as a **plain attribute**.

        It must be a real attribute (not a ``@property``) holding real sub-attributes, because
        IPython's completer evaluates the access chain through ``guarded_eval``, whose default
        ``'limited'`` policy rejects ``@property`` getters and ``__getattr__`` — so a dynamic
        chain never reaches ``__dir__`` and nothing tab-completes. Rebuilt after each
        `run_operation` so a newly-run op becomes completable. Discovers ops by scanning
        ``operations/``; layered on `get_operation_result`."""
        self.operation_results = OperationResultsNamespace(self)

    def rerun(self, op_cls: type[BaseOperation], **overrides):
        """Reconstruct an op from its stored params (+ overrides) and re-run it (forced).

        Works because dataclass fields == constructor args, so ``op_cls(**stored_params)``
        rebuilds the op."""
        rec = self._read_operation_record(op_cls.name)
        params = (rec.params if rec is not None else {}) or {}
        return self.run_operation(op_cls(**{**params, **overrides}), force=True)

    @property
    def repertoire_dir(self) -> Path:
        return self.db_dir / Layout.processed_dir.path

    def repertoire_meta(self, locus: Optional[str] = None) -> pl.DataFrame:
        """Per-locus repertoire metadata (Option C). ``locus=None`` returns the concatenation
        of every present locus (grain: one row per ``(repertoire_id, locus)``); a specific
        ``locus`` returns just that locus's table (one row per ``repertoire_id``). Callable, not
        a property, because a multi-locus dataset has no single default table."""
        loci = sorted(self._present_loci()) if locus is None else [locus]
        frames = [pl.read_parquet(self.db_dir / repertoire_meta_relpath(l)) for l in loci]
        df = frames[0] if len(frames) == 1 else pl.concat(frames)
        return df.select(_schema.REPERTOIRE_META.keys()).cast(_schema.REPERTOIRE_META)

    def full_repertoire_meta(self, locus: Optional[str] = None) -> pl.DataFrame:
        rep = self.repertoire_meta(locus)

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
    def operations(self) -> List[OperationRecord]:
        """All per-op records, discovered by scanning ``operations/*/operation.json``."""
        ops_dir = self.db_dir / Layout.operations_dir.path
        records = []
        if ops_dir.exists():
            for op_json in sorted(ops_dir.glob("*/operation.json")):
                records.append(OperationRecord.from_dict(json.loads(op_json.read_text())))
        return records

    def _repertoire_files(self, locus: Optional[str] = None) -> List[Path]:
        """Processed parquet paths (one per repertoire) for a locus, from the hive partitions
        ``*/locus={LOCUS}/*.parquet``. ``locus=None`` -> all present loci (excludes `_unassigned`);
        a specific locus (including ``"_unassigned"``) -> just that partition. The iterator owns
        locus filtering (E2)."""
        loci = sorted(self._present_loci()) if locus is None else [locus]
        files = []
        for l in loci:
            files.extend(sorted(self.db_dir.glob(loci_glob(l))))
        return files

    def read_repertoire(self, repertoire_id: str, locus: Optional[str] = None,
                        lazy: bool = False, filter_pass_only: bool = False):
        """Read one repertoire. ``locus=None`` hive-scans the whole repertoire dir and returns a
        ``locus`` column (recovered from the path) — the paired-chain access path (excludes the
        `_unassigned` bucket); a specific ``locus`` reads that one partition. Returns a
        DataFrame (or LazyFrame if ``lazy``)."""
        if locus is None:
            lf = pl.scan_parquet(self.db_dir / repertoire_dir_relpath(repertoire_id),
                                 hive_partitioning=True).filter(pl.col("locus") != UNASSIGNED)
        else:
            lf = pl.scan_parquet(self.db_dir / repertoire_locus_relpath(repertoire_id, locus))
        if filter_pass_only:
            lf = lf.filter(pl.col("filter_pass"))
        return lf if lazy else lf.collect(engine="streaming")

    def iter_repertoires(self, locus:Optional[str]=None, lazy=True, progress_bar=False, filter_pass_only=True, progress_desc:Optional[str]=None) -> Generator[str, pl.DataFrame | pl.LazyFrame]:
        files = self._repertoire_files(locus)
        if progress_bar:
            desc = progress_desc or "Iterating repertoires"
            progress = tqdm(total=len(files), desc=desc)

        for parquet_file in files:
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
            locus:Optional[str]=None,
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

        loci = sorted(self._present_loci()) if locus is None else [locus]

        for patient, patient_repertoires, in self.patient_meta.select("patient_id", "patient_repertoires").iter_rows():
            # Rebuild per-locus paths from the (unchanged) repertoire-id list; a repertoire only
            # has a file for the loci it actually produced, so keep the ones that exist.
            files = [
                self.db_dir / repertoire_locus_relpath(rep_id, l)
                for rep_id in patient_repertoires for l in loci
                if (self.db_dir / repertoire_locus_relpath(rep_id, l)).exists()
            ]
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
    
    def map_repertoires(
            self,
            fn,
            *,
            locus: Optional[str] = None,
            filter_pass_only: bool = True,
            lazy: bool = True,
            tag_repertoire_id: bool = True,
            tag_locus: bool = False,
            concat: bool = True,
            schema: Optional[pl.Schema] = None,
            progress_bar: bool = False,
            progress_desc: Optional[str] = None,
        ):
        """Map ``fn`` over each repertoire, tag ``repertoire_id`` (and optionally ``locus``), and
        concatenate. Consolidates the "iterate -> apply -> tag -> collect -> concat" loop shared
        by the per-repertoire ops, and is the single place the locus loop + locus tagging live
        (E2): ``locus=None`` spans all present loci, an op forwards the ``locus`` it was handed in
        ``_run(ds, locus)``.

        Each repertoire's (small) result is **collected eagerly, one at a time**, then the eager
        parts are concatenated — this bounds the working set to a single repertoire and does NOT
        build one lazy plan over every repertoire (which can explode memory for large sets).
        ``fn`` receives a lazy frame by default (``lazy=True``) so its own work stays lazy until
        the per-rep collect. Returns an eager ``DataFrame`` (or the list of eager parts when
        ``concat=False``). On an empty repertoire set, returns an empty frame if ``schema`` is
        given, else raises."""
        parts = []
        for rep_id, rep in self.iter_repertoires(
            locus=locus, lazy=lazy, filter_pass_only=filter_pass_only,
            progress_bar=progress_bar, progress_desc=progress_desc,
        ):
            part = fn(rep)
            tags = []
            if tag_repertoire_id:
                tags.append(pl.lit(rep_id).alias("repertoire_id"))
            if tag_locus:
                if locus is None:
                    raise ValueError("map_repertoires(tag_locus=True) requires a specific locus, not None.")
                tags.append(pl.lit(locus).alias("locus"))
            if tags:
                part = part.with_columns(*tags)
            if isinstance(part, pl.LazyFrame):
                part = part.collect(engine="streaming")   # per-repertoire collect: bounded memory
            parts.append(part)

        if not concat:
            return parts
        if not parts:
            if schema is not None:
                return pl.DataFrame(schema=schema)
            raise ValueError(
                f"map_repertoires produced no repertoires (locus={locus!r}); pass schema= to "
                f"get a typed-empty frame instead of raising."
            )
        return pl.concat(parts)

    @cached_property
    def n_repertoires(self):
        """Number of samples: unique ``repertoire_id`` across all loci (grain unchanged)."""
        return self.repertoire_meta().select(pl.col("repertoire_id").n_unique())[0, 0]

    @cached_property
    def n_patients(self):
        return self.patient_meta.select(pl.col("patient_id").n_unique())[0, 0]

    @cached_property
    def n_clonotypes(self):
        """Total passing clonotypes summed across every locus."""
        return self.repertoire_meta().select(pl.sum("n_clonotypes"))[0, 0]
    
    def __repr__(self):
        return f"TcrDataset \'{self.db_name}\', created_on {self.generation_meta['created_on']}, {self.n_clonotypes} clonotypes ({self.n_repertoires} repertoires, {self.n_patients} patients)."


class OperationResultsNamespace:
    """`ds.operation_results` — attribute access to operation outputs.

    Each successful op is set as a **real attribute** (an `OperationHandle`) at construction, so
    IPython's `guarded_eval` completer can traverse ``ds.operation_results.<op>...`` under its
    default ``'limited'`` policy (which rejects ``@property``/``__getattr__`` hops). `ds` rebuilds
    this namespace after each run (`_refresh_operation_results`), so new ops appear. `__getattr__`
    stays as a fallback for ops written after this namespace was built."""
    def __init__(self, ds: TcrDataset):
        self._ds = ds
        ops_dir = ds.db_dir / Layout.operations_dir.path
        if ops_dir.exists():
            for p in sorted(ops_dir.glob("*/operation.json")):
                op_name = p.parent.name
                if op_name.startswith("_"):
                    continue
                rec = ds._read_operation_record(op_name)
                if rec is not None and rec.status == "success":
                    setattr(self, op_name, OperationHandle(ds, rec))

    # NB: deliberately NO __getattr__ — its mere presence on the class makes IPython's
    # guarded_eval refuse to traverse this object during completion (limited mode), so the
    # chain never reaches `__dir__`. Ops are real attributes (above), refreshed after each run.

    def __dir__(self):
        ops_dir = self._ds.db_dir / Layout.operations_dir.path
        names = [p.parent.name for p in ops_dir.glob("*/operation.json")] if ops_dir.exists() else []
        return list(super().__dir__()) + names


class OperationHandle:
    """One operation's outputs. ``.<output>`` -> the frame (or `Path`); ``.<locus>.<output>`` for
    locus-aware ops.

    Loci are set as **real attributes** (sub-handles) so `guarded_eval` can traverse them during
    completion; outputs stay lazy (`__getattr__`) so no frame is loaded just to complete a name
    (the completer only ``dir()``s this handle, it never evaluates the leaf). ``__dir__`` lists
    outputs (+ loci) for tab-completion."""
    def __init__(self, ds: TcrDataset, rec: OperationRecord, locus: Optional[str] = None):
        object.__setattr__(self, "_ds", ds)
        object.__setattr__(self, "_rec", rec)
        object.__setattr__(self, "_locus", locus)
        if locus is None:
            for loc in self._loci():                      # real sub-handle per locus (completable)
                object.__setattr__(self, loc, OperationHandle(ds, rec, locus=loc))

    def _loci(self) -> List[str]:
        return sorted({o.locus for o in self._rec.outputs if o.locus is not None})

    def _output_names(self) -> List[str]:
        return sorted({o.name for o in self._rec.outputs
                       if self._locus is None or o.locus == self._locus})

    def __getattr__(self, x: str):
        if x.startswith("_"):
            raise AttributeError(x)
        if self._locus is None and x in self._loci():
            return OperationHandle(self._ds, self._rec, locus=x)
        if x in {o.name for o in self._rec.outputs}:
            return self._ds.get_operation_result(self._rec.operation_name, x, locus=self._locus)
        raise AttributeError(x)

    def __dir__(self):
        extra = self._output_names()
        if self._locus is None:
            extra = extra + self._loci()
        return list(super().__dir__()) + extra
