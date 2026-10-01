from __future__ import annotations
from functools import cached_property
import json
import logging
import warnings
from typing import Generator, List, Optional, Tuple
import polars as pl
from pathlib import Path
from tqdm import tqdm

from .operations.base import BaseOperation
from .operations.record import OPERATION_RECORD, OperationRecord
from .operations.runner import OperationRunner
from .grouper import Grouper
from .structure import (
    layout, Artifact, Handle, Store, Manifest, Migrator, DATASET_VERSION, required_dirs,
)
from .metadata import MetadataWriter
from .expressions import UNASSIGNED

log = logging.getLogger(__name__)


class LocusRequiredError(ValueError):
    """A locus-grain method called on a dataset with no locus bound.

    The price of one class instead of two. A `LocusDataset` would make "wrong grain" a
    method that does not exist; one class makes it a named exception, raised in the one line
    `_locus` shares among the methods that charge it — and spares the ten-odd hand-written
    forwards a second class needs to look like the first.
    """


class Dataset:
    """
    Handle to a repertoire dataset directory, optionally bound to one locus. Knows structure;
    provides accessors.

    Named `Dataset`, not `TcrDataset`: the library reads BCR loci as well as TR ones, and the
    locus set (`expressions.KNOWN_LOCI`) has covered both since the chain work. The old name
    described one of the two.

    Mostly read-through (every accessor reads from disk on demand). It is also the entry point
    for the four supported mutations — ``migrate`` (version upgrade), ``run_operation`` and
    ``rerun`` (generated results), and the ``metadata`` side-table writers — but it *implements*
    none of them: each constructs its writer (`Migrator`, `OperationRunner`, `MetadataWriter`)
    and hands off. One owner per mutation is the guarantee; a read-only dataset is not, and is
    not enforced — `store` is public and a `Handle` writes as well as reads.

    A bound locus is bound in the `Store`, so `{locus}` templates resolve with no call site
    passing one: `ds.select_locus("TRB").iter_repertoires()`. The locus-grain methods raise
    `LocusRequiredError` unbound; the rest work either way.
    """

    def __init__(self, db_dir:str|Path, *, locus: Optional[str] = None):
        self.db_dir = Path(db_dir)
        self.db_name = self.db_dir.name
        # Normally set by `select_locus`; `run_operation` binds one dataset per pass.
        self.locus = locus
        self._validate_dirs()
        self._check_version()

    @classmethod
    def migrated(cls, db_dir:str|Path) -> "Dataset":
        """Alternative constructor: open a dataset, upgrading it **in place** to the current
        version first. Unlike ``Dataset(path)`` (which only warns when behind), this mutates
        the dataset on disk and returns a handle at ``DATASET_VERSION``.

        The upgrade runs *before* any dataset exists, which is the whole reason `Migrator` is a
        separate class: on a v0 tree the constructor's own checks would reject the very
        directory it is about to fix."""
        Migrator(db_dir).migrate()
        return cls(db_dir)

    @property
    def store(self) -> Store:
        """Every path in this dataset resolves through here. A bound locus is bound here, so
        `select_locus` *is* `with_params(locus=...)` and no read has to pass one."""
        store = Store(self.db_dir)
        return store.with_params(locus=self.locus) if self.locus else store

    def path(self, artifact, **params) -> Path:
        """Absolute path of an Artifact within this dataset. Template parameters
        (``locus``, ``repertoire_id``) are passed as keywords."""
        return self.store(artifact, **params).path()

    @cached_property
    def _manifest(self) -> Manifest:
        return Manifest.read(self.store)

    @property
    def version(self) -> int:
        return self._manifest.version

    def _validate_dirs(self):
        if not self.db_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.db_dir}")

        # Same derivation ingestion builds the tree from, so the two cannot disagree about
        # which directories a dataset must have.
        missing = [str(d.relative_to(self.db_dir))
                   for d in required_dirs(self.db_dir, layout.ARTIFACTS) if not d.is_dir()]
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
                    f"Open with Dataset.migrated(path) to upgrade it in place.",
                    stacklevel=2,
                )
            case 1:
                raise RuntimeError(
                    f"Dataset is v{disk} but this tcrio only understands v{lib}. Upgrade tcrio."
                )

    def migrate(self, target: int = DATASET_VERSION, *, dry_run: bool = False) -> list:
        """Upgrade this dataset in place. Delegates; the dataset's only stake is that its
        cached manifest is now stale."""
        plan = Migrator(self.db_dir).migrate(target, dry_run=dry_run)
        self.__dict__.pop("_manifest", None)
        return plan

    # --- locus ---------------------------------------------------------------------------

    @cached_property
    def dispatch_loci(self) -> frozenset:
        """Every locus partition on disk — one listing of `processed_repertoires/locus=*/`.

        This used to be a manifest field copied in at ingestion, which is a derived value
        persisted: delete a locus's directory and the dataset went on claiming it. The listing
        cannot disagree with the tree because it *is* the tree, and the locus-first layout is
        what made it one directory read instead of a glob across every repertoire.
        """
        return frozenset(d.name.split("=", 1)[1]
                         for d in Store(self.db_dir)(layout.LOCUS_DIR).glob())

    @property
    def present_loci(self) -> frozenset:
        """The real loci — `dispatch_loci` without the `_unassigned` QC bucket, which is where
        rows with no derivable locus go, not a locus. Drives the fan-out in `run_operation`."""
        return self.dispatch_loci - {UNASSIGNED}

    def select_locus(self, locus: str) -> "Dataset":
        """This dataset bound to one locus. Every locus-grain read then needs no argument."""
        if locus not in self.dispatch_loci:
            raise ValueError(
                f"Locus {locus!r} is not in this dataset; have {sorted(self.dispatch_loci)}."
            )
        return Dataset(self.db_dir, locus=locus)

    def each_locus(self) -> Generator["Dataset", None, None]:
        """One bound dataset per present locus — the answer to "do X for every locus". The
        concat is one line and belongs to whoever wants the frame, not to this iterator."""
        for l in sorted(self.present_loci):
            yield Dataset(self.db_dir, locus=l)

    def _locus(self, method: str) -> str:
        """The bound locus, or a named error. The one line every locus-grain method shares."""
        if self.locus is None:
            raise LocusRequiredError(
                f"{method}() is locus-grain; call it on ds.select_locus(<LOCUS>). "
                f"Present loci: {sorted(self.present_loci)}."
            )
        return self.locus

    # --- operations -----------------------------------------------------------------

    def run_operation(self, operation: BaseOperation, *, force: bool = False):
        """Run an operation over the loci it supports. Delegates: everything about fan-out,
        staleness and records lives in `OperationRunner`, which is where an operation's own
        code can never reach it."""
        return OperationRunner(self, force=force).run(operation)

    def rerun(self, op_cls: type[BaseOperation], **overrides):
        """Re-run an operation from its recorded config, with overrides. Forced."""
        return OperationRunner(self).rebuild(op_cls, **overrides)

    def result(self, art: Artifact, *, locus: Optional[str] = None, **params) -> Handle:
        """A handle on one operation output.

        The artifact is the operation's own class attribute, so the call names both at once
        and a typo is an AttributeError:

            ds.result(DiversityReport.diversity_summary, locus="TRB").read()

        The output directory comes from `art.owner.name` — which is why `BaseOperation.
        artifacts()` refuses an inherited artifact. `locus` defaults to the bound one, so on
        `ds.select_locus("TRB")` it can be left off entirely."""
        locus = locus if locus is not None else self._locus("result")
        if not isinstance(art, Artifact):
            raise TypeError(
                f"result() takes an operation's Artifact, not {art!r}. Import the operations "
                f"namespace and name the output — `from tcr_io import operations as op` then "
                f"`ds.result(op.FilteringReport.filter_summary)`, which autocompletes and casts. "
                f"To read by name instead, use raw_result(op_name, output).")
        if art.owner is None:
            raise ValueError(f"{art!r} is not declared on an operation class; use raw_result().")
        return self._op_store(art.owner.name, locus)(art, **params)

    def raw_result(self, op_name: str, output: str, *, locus: Optional[str] = None) -> Handle:
        """The same read resolved through the record alone — by strings, no class import.

        For datasets produced by a library version whose operation classes you do not have.
        The record does not store schemas, so this read is uncast."""
        rec = self._record(op_name, locus if locus is not None else self._locus("raw_result"))
        if rec is None or rec.status != "success":
            raise ValueError(f"No successful result for {op_name!r} (locus={locus!r}).")
        matches = [o for o in rec.outputs if o.name == output]
        if len(matches) != 1:
            raise ValueError(
                f"{op_name!r} has {len(matches)} outputs named {output!r}; "
                f"available: {sorted({o.name for o in rec.outputs})}"
            )
        return self._op_store(op_name, rec.locus)(matches[0].artifact())

    def _op_store(self, op_name: str, locus: str) -> Store:
        """Root of one (operation, locus) pass — the same binder the runner writes through.

        Built from `self.store`, not from `Store(self.db_dir)`. Every store in the library now
        descends from the dataset's, so there is one place a root is chosen. `with_root` keeps
        the bound parameters, and the explicit `locus=` then pins the one that matters — so a
        locus-bound dataset and an unbound one resolve the same path."""
        return self.store.with_root(layout.OPERATIONS_DIR, op_name, locus) \
                         .with_params(locus=locus)

    def _record(self, op_name: str, locus: str) -> Optional[OperationRecord]:
        return OperationRecord.read(self._op_store(op_name, locus)(OPERATION_RECORD))

    @property
    def repertoire_dir(self) -> Path:
        return self.path(layout.PROCESSED_DIR)

    @property
    def repertoire_meta(self) -> pl.DataFrame:
        """One row per repertoire: source files, patient. A property again, and locus-blind —
        none of it varies by locus, so there is one table and the binding does not enter."""
        return self.store(layout.REPERTOIRE_META).read()

    @property
    def full_repertoire_meta(self) -> pl.DataFrame:
        rep = self.repertoire_meta

        extra = self.store(layout.REPERTOIRE_EXTRA).read()
        return rep if extra is None else rep.join(extra, on="repertoire_id", how="left")

    @property
    def repertoire_counts(self) -> pl.DataFrame:
        """Clonotype counts, one row per repertoire.

        The other half of the old single repertoire-meta table, kept separate because it is the
        half that actually depends on the locus. Bound to a locus: that locus's counts, read
        straight off its shard. Unbound: the same columns summed across every present locus —
        still one row per repertoire, whichever loci it appears in.

        Both arms return the SAME schema, which is what lets this be a property. It used to take
        a `locus` argument and, unbound, concatenate the per-locus tables with an added `locus`
        column — so an unbound read handed back a different shape *and* a different grain than a
        bound one, and every caller had to know which it was holding. The per-locus breakdown is
        now asked for by binding: `ds.select_locus("TRB").repertoire_counts`.
        """
        if self.locus:
            return self.store(layout.REPERTOIRE_COUNTS, locus=self.locus).read()
        frames = [self.store(layout.REPERTOIRE_COUNTS, locus=l).read()
                  for l in sorted(self.present_loci)]
        # A dataset can legitimately hold no real locus — every row landed in `_unassigned`,
        # which is what a wrong reader looks like. That is an empty table, not an error, and
        # returning it typed keeps `n_clonotypes` (and so `__repr__`) working on the tree that
        # most needs looking at.
        if not frames:
            return pl.DataFrame(schema=layout.REPERTOIRE_COUNTS.schema)
        # Summed columns come from the artifact's schema, so a count added there is carried here
        # without a second list to keep in step. Everything but the key is a count.
        return (pl.concat(frames)
                  .group_by("repertoire_id")
                  .agg([pl.col(c).sum() for c in layout.REPERTOIRE_COUNTS.schema
                        if c != "repertoire_id"])
                  .sort("repertoire_id"))

    @property
    def patient_meta(self) -> pl.DataFrame:
        return self.store(layout.PATIENT_META).read()

    @property
    def full_patient_meta(self) -> pl.DataFrame:
        pat = self.patient_meta

        for side in (layout.PATIENT_EXTRA, layout.HLA):
            df = self.store(side).read()
            if df is not None:
                pat = pat.join(df, on="patient_id", how="left")
        return pat

    @property
    def metadata(self) -> MetadataWriter:
        """The metadata write API: `ds.metadata.set_patient(df)`, `.set_repertoire`,
        `.set_publication`, `.set_hla`. A separate object because writing metadata shares a
        validation path with nothing on the read side."""
        return MetadataWriter(self)

    @cached_property
    def publication_meta(self) -> dict:
        return self.store(layout.PUBLICATION_IDS).read().to_dict(as_series=False)

    @cached_property
    def full_publication_meta(self) -> pl.DataFrame:
        pub = pl.DataFrame(self.publication_meta)

        extra = self.store(layout.PUBLICATION_EXTRA).read()
        return pub if extra is None else pub.join(extra, on="publication_id", how="left")

    @cached_property
    def generation_meta(self) -> dict:
        """How this dataset was ingested: source, mappers, reader, library version, and the
        filter set applied per locus. Written once at ingestion; see
        `DatasetIngester._generate_generation_metadata` for what each field is worth."""
        return self.store(layout.GENERATION_META).read()

    @property
    def operations(self) -> List[OperationRecord]:
        """Every operation record in this dataset, one per (operation, locus).

        One glob: every record sits at `operations/<op>/<locus>/operation.json`, because every
        operation is locus-grain."""
        ops_dir = self.db_dir / layout.OPERATIONS_DIR
        return [OperationRecord.from_dict(json.loads(p.read_text()))
                for p in sorted(ops_dir.glob("*/*/operation.json"))] if ops_dir.exists() else []

    @property
    def clonotypes(self) -> pl.LazyFrame:
        """Every repertoire's passing clonotypes, concatenated, as one lazy frame.

        Bound to a locus, this is that locus's shards read as a single frame: `repertoire_id`
        is left unbound, so the handle globs `locus={LOCUS}/*.parquet` and scans the lot.
        Unbound, it is the same per present locus, each part tagged with the locus it came
        from — tagged *after* the schema cast, which is why this needs neither hive
        partitioning nor a `locus` column in `REPERTOIRE`.

        `_unassigned` is never included: it is not a locus, and its rows do not pass.

        Passing rows only, which is what the name promises — a clonotype is a row that
        survived ingest filtering. The unfiltered frame is `ds.store(layout.REPERTOIRE_FILE).scan()`
        and the per-repertoire loop is `iter_repertoires`; this is the one that answers "all
        of it, in one frame".
        """
        if self.locus:
            return self.store(layout.REPERTOIRE_FILE).scan().filter(pl.col("filter_reason").is_null())
        frames = [d.clonotypes.with_columns(pl.lit(d.locus).alias("locus"))
                  for d in self.each_locus()]
        # A dataset can legitimately hold no real locus — every row landed in `_unassigned`,
        # which is what a wrong reader looks like. Typed-empty, for the same reason
        # `repertoire_counts` returns one: the tree that most needs looking at should still
        # answer questions about its own shape.
        if not frames:
            return pl.LazyFrame(schema={**layout.REPERTOIRE_FILE.schema, "locus": pl.Utf8})
        return pl.concat(frames)

    def _repertoire_files(self) -> List[Path]:
        """LOCUS-GRAIN. Processed parquet paths (one per repertoire) for the bound locus, from
        the hive partitions ``*/locus={LOCUS}/*.parquet``."""
        self._locus("_repertoire_files")
        return self.store(layout.REPERTOIRE_FILE).glob()

    def read_repertoire(self, repertoire_id: str, lazy: bool = False,
                        passing_only: bool = False):
        """LOCUS-GRAIN. One repertoire's clonotypes on the bound locus. Returns a DataFrame
        (or LazyFrame if ``lazy``)."""
        self._locus("read_repertoire")
        lf = self.store(layout.REPERTOIRE_FILE,
                        repertoire_id=layout.safe_repertoire_name(repertoire_id)).scan()
        if passing_only:
            lf = lf.filter(pl.col("filter_reason").is_null())
        return lf if lazy else lf.collect(engine="streaming")

    def iter_repertoires(self, lazy=True, progress_bar=False, passing_only=True, progress_desc:Optional[str]=None) -> Generator[str, pl.DataFrame | pl.LazyFrame]:
        """LOCUS-GRAIN. Each repertoire on the bound locus, as ``(repertoire_id, frame)``."""
        files = self._repertoire_files()
        if progress_bar:
            desc = progress_desc or "Iterating repertoires"
            progress = tqdm(total=len(files), desc=desc)

        for parquet_file in files:
            repertoire_id = pl.scan_parquet(parquet_file).select(pl.col("repertoire_id").first()).collect()[0, 0]
            df = pl.scan_parquet(parquet_file)
            if passing_only:
                df = df.filter(pl.col("filter_reason").is_null())
            if lazy:
                yield repertoire_id, df
            else:
                yield repertoire_id, df.collect(engine="streaming")
            if progress_bar:
                progress.update(1)

    def iter_repertoires_by_patient(
            self,
            deduplicate=True,
            passing_only=True,
            lazy=True,
            progress_bar=False,
            progress_desc:Optional[str]=None
            ) -> Generator[Tuple[str, pl.DataFrame | pl.LazyFrame], None, None]:
        """LOCUS-GRAIN, and patient-keyed. A cross-locus concat of clonotypes is not a thing
        anyone wants, so this needs the binding like the rest of the repertoire reads."""
        self._locus("iter_repertoires_by_patient")
        patient_meta = self.full_patient_meta

        if progress_bar:
            desc = progress_desc or "Iterating repertoires by patient"
            progress = tqdm(total=patient_meta.select(pl.col("patient_id").n_unique())[0, 0], desc=desc)

        for patient, patient_repertoires, in self.patient_meta.select("patient_id", "patient_repertoires").iter_rows():
            # Rebuild paths from the (unchanged) repertoire-id list; a repertoire only has a file
            # for the loci it actually produced, so keep the ones that exist.
            files = [
                f for rep_id in patient_repertoires
                for f in self.store(layout.REPERTOIRE_FILE,
                                    repertoire_id=layout.safe_repertoire_name(rep_id)).glob()
            ]
            n_files = len(files)

            df = pl.concat([pl.scan_parquet(f).with_columns(file=pl.lit(f.name)) for f in files])
            if passing_only:
                df = df.filter(pl.col("filter_reason").is_null())

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
            passing_only: bool = True,
            lazy: bool = True,
            tag_repertoire_id: bool = True,
            concat: bool = True,
            schema: Optional[pl.Schema] = None,
            progress_bar: bool = False,
            progress_desc: Optional[str] = None,
        ):
        """LOCUS-GRAIN. Map ``fn`` over each repertoire on the bound locus, tag
        ``repertoire_id``, and concatenate. Consolidates the "iterate -> apply -> tag -> collect
        -> concat" loop shared by the per-repertoire ops. No locus tagging: every row came from
        the one bound locus, so the column would be a constant the caller already knows.

        Each repertoire's (small) result is **collected eagerly, one at a time**, then the eager
        parts are concatenated — this bounds the working set to a single repertoire and does NOT
        build one lazy plan over every repertoire (which can explode memory for large sets).
        ``fn`` receives a lazy frame by default (``lazy=True``) so its own work stays lazy until
        the per-rep collect. Returns an eager ``DataFrame`` (or the list of eager parts when
        ``concat=False``). On an empty repertoire set, returns an empty frame if ``schema`` is
        given, else raises."""
        parts = []
        for rep_id, rep in self.iter_repertoires(
            lazy=lazy, passing_only=passing_only,
            progress_bar=progress_bar, progress_desc=progress_desc,
        ):
            part = fn(rep)
            if tag_repertoire_id:
                part = part.with_columns(pl.lit(rep_id).alias("repertoire_id"))
            if isinstance(part, pl.LazyFrame):
                part = part.collect(engine="streaming")   # per-repertoire collect: bounded memory
            parts.append(part)

        if not concat:
            return parts
        if not parts:
            if schema is not None:
                return pl.DataFrame(schema=schema)
            raise ValueError(
                f"map_repertoires produced no repertoires (locus={self.locus!r}); pass schema= to "
                f"get a typed-empty frame instead of raising."
            )
        return pl.concat(parts)

    @cached_property
    def n_repertoires(self):
        """Number of samples. One row per repertoire in `repertoire_meta` now, so this is a
        height — the `n_unique()` it replaces was deduplicating the per-locus concat."""
        return self.repertoire_meta.height

    @cached_property
    def n_patients(self):
        return self.patient_meta.select(pl.col("patient_id").n_unique())[0, 0]

    @cached_property
    def n_clonotypes(self):
        """Total passing clonotypes: the bound locus's, or summed over every locus."""
        return self.repertoire_counts.select(pl.sum("n_clonotypes"))[0, 0]
    
    def __repr__(self):
        bound = f" [locus={self.locus}]" if self.locus else ""
        return f"""Processed Dataset \'{self.db_name}\'{bound}, dataset v{self.version}, 
        present loci {sorted(self.present_loci)},
        created_on {self.generation_meta['created_on']}, 
        {self.n_clonotypes} clonotypes ({self.n_repertoires} repertoires, {self.n_patients} patients),
        """
