from collections import defaultdict
from tcr_io.mappers import BaseMapper
from tcr_io.readers import BaseReader, ReaderFactory
from typing import List
import logging
from tqdm import tqdm
from pathlib import Path
import polars as pl
from datetime import datetime
import shutil


from .dataset import Dataset
from ._internal import __version__ as _tcrio_version
from .expressions import assign_locus, UNASSIGNED
from .filters import FilterSet, PerLocusFilterSet
from .structure import layout, Store, Manifest
from .structure import schema as s


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _process_path_wildcards(f: Path|str) -> List[Path]:
    path = Path(f)
    if path.is_file():
        return [path]
    elif path.is_dir():
        return list(path.iterdir())
    elif "*" in f.name:
        return list(path.parent.glob(path.name))
    else:
        raise ValueError(f"Path {f} is not a file, directory, or wildcard pattern.")

class DatasetIngester:

    def __init__(
        self,
        db_dir: str | Path,
        db_name: str,
        repertoire_mapper: BaseMapper,
        patient_mapper: BaseMapper,
        reader: BaseReader | ReaderFactory | None = None,
        publication_ids : List[str] | None = None,
        allow_overwrite: bool = False,
        filter_set: FilterSet | PerLocusFilterSet | str | None = None,
    ):
        self.db_dir = (Path(db_dir) / db_name).resolve()
        self.store = Store(self.db_dir)      # every path this ingester writes resolves through here
        self.reader = reader or ReaderFactory()
        self.repertoire_mapper = repertoire_mapper
        self.patient_mapper = patient_mapper
        self.db_name = db_name
        self.publication_ids = publication_ids or []
        self.allow_overwrite = allow_overwrite
        # Quality filters run AFTER locus assignment (locus-aware). A preset NAME is the
        # intended interface — it is the one thing in the ingest record that re-runs — so it
        # is accepted directly; a dataset needing different cleaning gets a preset registered
        # rather than a FilterSet assembled here. A plain FilterSet is applied uniformly to
        # every known locus; None -> the default TR preset.
        if filter_set is None:
            filter_set = "default_trb"
        if isinstance(filter_set, str):
            filter_set = FilterSet.named(filter_set)
        if isinstance(filter_set, FilterSet):
            filter_set = PerLocusFilterSet.uniform(filter_set)
        self.filter_set = filter_set

    def _create_structure(self):
        if self.db_dir.exists():
            if not self.allow_overwrite:
                raise FileExistsError(
                    f"Dataset already exists: {self.db_dir}."
                )
            else:
                logger.warning(f"Dataset directory {self.db_dir} already exists and will be overwritten.")
                self._delete_existing_data() # clears processed_repertoires + stale operation outputs

        # The directories every dataset must have, derived from the artifact table rather
        # than listed here — so a new artifact cannot land without its home.
        self.store.create_tree(layout.ARTIFACTS)

        
    def _create_mapping(self, dir:Path|str):

        dir : List[Path] = _process_path_wildcards(dir)

        data, skipped_files = self._resolve_dir(dir)
        repertoires = defaultdict(list)
        repertoires_to_patients = dict()

        for file, repertoire_id, patient_id in data:
            repertoires[repertoire_id].append(file)
            repertoires_to_patients[repertoire_id] = patient_id

        return repertoires, repertoires_to_patients, skipped_files

    def _resolve_dir(self, files: List[Path]):
        data = []
        skipped_files = []
        for file in files:
            if file.is_dir():
                continue    
            try:
                repertoire_id = self.repertoire_mapper.transform(file)
                patient_id = self.patient_mapper.transform(file)
                data.append((file, repertoire_id, patient_id))

            except KeyError as e:
                skipped_files.append(file)

        return data, skipped_files
    
    def run(self, dir:Path|str) -> Dataset:
        logger.info(f"Processing dataset {self.db_name} from {dir}...")
        self._create_structure()
        logger.info("Creating mappings...")
        repertoires, repertoires_to_patients, skipped = self._create_mapping(dir)
        logger.info(f"Mapping created. {len(repertoires)} repertoires mapped from {len(set(repertoires_to_patients.values()))} unique patients, {len(skipped)} files skipped.")
        present_loci = self._process_repertoires(repertoires)   # loci actually written (excl. _unassigned)
        self._generate_repertoire_metadata(repertoires, repertoires_to_patients, present_loci)
        self.store(layout.PATIENT_META).write(self._generate_patient_metadata(repertoires_to_patients))
        self.store(layout.PUBLICATION_IDS).write(self._generate_publication_metadata())
        self.store(layout.GENERATION_META).write(self._generate_generation_metadata(dir))
        # operations/ is created by the first operation that writes into it; ops fill it
        # on demand with per-op operation.json records — there is no global ledger to seed.
        # The manifest records what cannot be read off the tree, and that is the version and
        # nothing else. Which loci a dataset holds can be read — `processed_repertoires/locus=*/`
        # — and how it was ingested is the ingest record's job, one line up.
        Manifest.current().write(self.store)

        logger.info("Dataset processing complete.")
        return Dataset(self.db_dir)
    
    def _process_repertoires(self, repertoires) -> set:
        """Write each repertoire's rows into the locus-first tree
        (``processed_repertoires/locus={LOCUS}/{id}.parquet``); rows with no derivable locus ride
        the ``locus=_unassigned/`` partition (see `assign_locus`).
        ``clonotype_id`` is a per-``(repertoire, locus)`` row number (``int_range().over("locus")``),
        so identity is ``(repertoire_id, locus, clonotype_id)`` — the hive grain.

        Bulk streams in one `sink_parquet(PartitionBy)` pass (no collect). Single-cell (a
        ``cell_id: List`` column from the grouper) **collects once** so one fixed row order backs
        both the clonotype table and the exploded ``clone_to_cell`` map — two lazy executions could
        reorder and desync the positional id (see docs/clonotype_id_implementation_plan.md §3).
        Returns the set of real loci written across all repertoires (excluding `_unassigned`)."""
        present: set = set()
        pbar = tqdm(repertoires.items(), desc="Reading repertoires")

        for repertoire_id, files in pbar:
            df = self.reader.run(files).with_columns(
                pl.lit(repertoire_id).alias("repertoire_id"),
                assign_locus(),                          # partition key; unknown/mismatch -> _unassigned
            ).pipe(                                       # locus-aware quality filter ->
                self.filter_set.run                      # filter_reason (null == passed)
            )

            safe_id = layout.safe_repertoire_name(repertoire_id)
            written: set = set()
            partition = pl.PartitionBy(
                # One base for the whole dataset: the repertoire is the leaf filename now, not a
                # directory level, so polars is told where the tree starts and the provider says
                # where each file lands. (polars enforces that the provider's path start here.)
                str(self.store(layout.PROCESSED_DIR).path()), key=["locus"], include_key=False,
                file_path_provider=self._locus_file_provider(repertoire_id, written),
            )
            clonotype_id = pl.int_range(pl.len(), dtype=pl.UInt32).over("locus")

            if df.collect_schema().get("cell_id") == pl.List(pl.Utf8):
                # single-cell: collect ONCE, then both artifacts share this row order / id assignment
                grouped = df.collect(engine="streaming").with_columns(clonotype_id=clonotype_id)
                grouped.select([*s.REPERTOIRE.keys(), "locus"]).write_parquet(partition)

                # lazy: meta/clone_to_cell exists only for single-cell (write() mkdir -p's it)
                self.store(layout.CLONE_TO_CELL, repertoire_id=safe_id).write(
                    grouped.select("repertoire_id", "locus", "clonotype_id", "cell_id")
                           .explode("cell_id"))
            else:
                # bulk: fully streamed single sink; clonotype_id numbered within each locus partition
                (df.with_columns(clonotype_id=clonotype_id)
                   .select([*s.REPERTOIRE.keys(), "locus"])
                   .sink_parquet(partition))

            present |= written

            # update pbar with reader
            pbar.set_postfix_str(
                f"Reader: {self.reader.name}, {"+".join(sorted(written - {UNASSIGNED}))}")

        present.discard(UNASSIGNED)
        return present

    def _locus_file_provider(self, repertoire_id, written: set):
        """`file_path_provider` for `PartitionBy`: name each partition's file `{id}.parquet` under
        its `locus={value}/` hive dir (rather than polars' `00000000.parquet`).

        It is also the only thing that can say which loci THIS repertoire produced, and adds each
        to `written` as it goes: the partitions all share one base directory now, so a
        `locus=*` listing would answer for every repertoire ingested so far.
        """
        def provider(args):
            locus = args.partition_keys.item()          # the single partition value for this file
            written.add(locus)
            out = self.store(layout.REPERTOIRE_FILE,
                             repertoire_id=layout.safe_repertoire_name(repertoire_id),
                             locus=locus).path()
            out.parent.mkdir(parents=True, exist_ok=True)
            return out
        return provider

    def _generate_repertoire_metadata(self, repertoires, repertoires_to_patients, present_loci) -> None:
        """Two grains, two tables.

        `meta/repertoire/repertoire.parquet` is what is true of a repertoire whichever locus you
        look at — its source files, its patient — written once. `locus={L}/counts.parquet` holds
        the counts, which mean nothing outside a locus, and exists only for the loci that
        repertoire actually produced. The single per-locus table this replaces copied the
        invariant half into every locus, so a patient_id fix had to be applied seven times.
        """
        self.store(layout.REPERTOIRE_META).write(pl.DataFrame({
            "repertoire_id": list(repertoires.keys()),
            "source_files": [[f.name for f in files] for files in repertoires.values()],
            "patient_id": [repertoires_to_patients[rep_id] for rep_id in repertoires],
        }))

        for locus in sorted(present_loci):   # present_loci already excludes _unassigned
            counts = [
                pl.scan_parquet(f).select(
                    pl.first("repertoire_id"),
                    pl.col("filter_reason").is_null().sum().alias("n_clonotypes"),
                    pl.col("filter_reason").is_not_null().sum().alias("n_filtered_clonotypes"),
                    pl.sum("duplicate_count").alias("total_duplicates"),
                )
                for f in self.store(layout.REPERTOIRE_FILE, locus=locus).glob()
            ]
            if not counts:
                continue
            self.store(layout.REPERTOIRE_COUNTS, locus=locus).write(
                pl.concat(counts).collect(engine="streaming"))

    def _generate_patient_metadata(self, repertoires_to_patients):

        repertoires_by_patient = defaultdict(list)
        for rep_id, patient_id in repertoires_to_patients.items():
            repertoires_by_patient[patient_id].append(rep_id)
        
        df = pl.DataFrame({
            "patient_id": list(repertoires_by_patient.keys()),
            "patient_repertoires": list(repertoires_by_patient.values()),
            "n_repertoires": [len(repertoires) for repertoires in repertoires_by_patient.values()]
        })

        return df
    

    def _generate_publication_metadata(self):
        df = pl.DataFrame({
            "publication_id": self.publication_ids
        })

        return df
    
    def _generate_generation_metadata(self, data_dir: Path) -> dict:
        """How this dataset was made — the only record of it.

        Ingestion is lossy: the reader normalises columns and the filter set marks rows as
        excluded, both in place. Nothing downstream can recover what was done, and
        `filter_reason` records only the filters that FIRED — a filter that rejected nothing
        leaves no trace in the data. So the set applied is written here, per locus.

        `filters` carries both a preset name and the names it expanded to at write time:
        the preset is what re-runs, and the expansion stays readable after a later release
        changes what that preset means. `tcrio_version` is what says which release that was.

        The mappers and reader are `repr` strings — documentation, not deserialisable. The
        preset name is the one field in here that reproduces rather than describes.
        """
        return {
            "dataset_name": self.db_name,
            "created_on": datetime.now().date().isoformat(),
            "source": str(data_dir),
            "tcrio_version": _tcrio_version,
            "reader": repr(self.reader),
            "repertoire_mapper": repr(self.repertoire_mapper),
            "patient_mapper": repr(self.patient_mapper),
            "filters": self.filter_set.provenance(),
        }
    
    def _delete_existing_data(self):
        # Re-ingesting the source data makes every derived operation output stale (its inputs
        # may have changed), so clear operations/ alongside processed_repertoires. Both are
        # not recreated here; ops recompute on demand and mkdir their own output dir.
        shutil.rmtree(self.store(layout.PROCESSED_DIR).path())
        ops_dir = self.db_dir / layout.OPERATIONS_DIR
        if ops_dir.exists():
            shutil.rmtree(ops_dir)
            logger.warning(
                "Cleared existing operation results (operations/) — inputs changed on "
                "re-ingestion; rerun operations to regenerate."
            )
        
    def _test_mappers(self, dir:Path|str):
        dir : List[Path] = _process_path_wildcards(dir)
        print("Repertoire mapping : ")
        print("-------------------")
        self.repertoire_mapper.test(dir)
        print("\nPatient mapping : ")
        print("-------------------")
        self.patient_mapper.test(dir)