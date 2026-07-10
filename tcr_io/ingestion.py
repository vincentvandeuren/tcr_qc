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


from .dataset import TcrDataset
from .expressions import assign_locus, UNASSIGNED
from .structure import (
    Layout, REQUIRED_DIRS, GENERATED_DIRS, repertoire_dir_relpath, repertoire_locus_relpath,
    loci_glob, repertoire_meta_relpath, Manifest,
    REPERTOIRE, REPERTOIRE_META, PATIENT_META, GENERATION_META, PUBLICATION_META,
)


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
        allow_overwrite: bool = False
    ):
        self.db_dir = (Path(db_dir) / db_name).resolve()
        self.reader = reader or ReaderFactory()
        self.repertoire_mapper = repertoire_mapper
        self.patient_mapper = patient_mapper
        self.db_name = db_name
        self.publication_ids = publication_ids or []
        self.allow_overwrite = allow_overwrite

    def _create_structure(self):
        if self.db_dir.exists():
            if not self.allow_overwrite:
                raise FileExistsError(
                    f"Dataset already exists: {self.db_dir}."
                )
            else:
                logger.warning(f"Dataset directory {self.db_dir} already exists and will be overwritten.")
                self._delete_existing_data() # clears processed_repertoires + stale operation outputs

        for subdir in REQUIRED_DIRS + GENERATED_DIRS:
            (self.db_dir / subdir).mkdir(parents=True, exist_ok=True)

        
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
    
    def run(self, dir:Path|str) -> TcrDataset:
        logger.info(f"Processing dataset {self.db_name} from {dir}...")
        self._create_structure()
        logger.info("Creating mappings...")
        repertoires, repertoires_to_patients, skipped = self._create_mapping(dir)
        logger.info(f"Mapping created. {len(repertoires)} repertoires mapped from {len(set(repertoires_to_patients.values()))} unique patients, {len(skipped)} files skipped.")
        present_loci = self._process_repertoires(repertoires)   # loci actually written (excl. _unassigned)
        self._generate_repertoire_metadata(repertoires, repertoires_to_patients, present_loci)
        self._generate_patient_metadata(repertoires_to_patients).write_parquet(
            self.db_dir / Layout.patient_meta.path
        )
        self._generate_publication_metadata().write_ndjson(
            self.db_dir / Layout.publication_ids.path
        )
        self._generate_generation_metadata(dir).write_ndjson(
            self.db_dir / Layout.generation_meta.path
        )
        # operations/ is created empty by _create_structure (a GENERATED_DIR); ops fill it
        # on demand with per-op operation.json records — there is no global ledger to seed.
        Manifest.current(present_loci=sorted(present_loci)).write(self.db_dir / Layout.manifest.path)

        logger.info("Dataset processing complete.")
        return TcrDataset(self.db_dir)
    
    def _process_repertoires(self, repertoires) -> set:
        """Write each repertoire in a single streaming `pl.PartitionBy` pass, partitioned by locus.
        A repertoire's frame is derived once (read/dedup/filter) and streamed to
        ``{id}/locus={LOCUS}/{id}.parquet`` — no collect, no per-locus re-execution. Rows with no
        derivable locus ride the ``locus=_unassigned/`` partition (see `assign_locus`). Returns the
        set of real loci written across all repertoires (excluding `_unassigned`)."""
        present: set = set()
        for repertoire_id, files in tqdm(repertoires.items(), desc="Reading repertoires"):
            df = self.reader.run(files).with_columns(
                pl.lit(repertoire_id).alias("repertoire_id"),
                assign_locus(),                          # partition key; unknown/mismatch -> _unassigned
            ).select([*REPERTOIRE.keys(), "locus"])       # locus is the key only (include_key=False)

            base = self.db_dir / repertoire_dir_relpath(repertoire_id)
            df.sink_parquet(pl.PartitionBy(
                str(base), key=["locus"], include_key=False,
                file_path_provider=self._locus_file_provider(repertoire_id),
            ))
            present |= self._loci_written(base)
        present.discard(UNASSIGNED)
        return present

    def _locus_file_provider(self, repertoire_id):
        """`file_path_provider` for `PartitionBy`: name each partition's file `{id}.parquet` under
        its `locus={value}/` hive dir (rather than polars' `00000000.parquet`)."""
        def provider(args):
            locus = args.partition_keys.item()          # the single partition value for this file
            out = self.db_dir / repertoire_locus_relpath(repertoire_id, locus)
            out.parent.mkdir(parents=True, exist_ok=True)
            return out
        return provider

    @staticmethod
    def _loci_written(base) -> set:
        """Locus values a repertoire produced, from its `locus=<v>/` partition dirs."""
        return {p.name.split("=", 1)[1] for p in base.glob("locus=*") if p.is_dir()}

    def _generate_repertoire_metadata(self, repertoires, repertoires_to_patients, present_loci) -> None:
        """Write one `repertoire_meta` parquet per locus (Option C). Each keeps the
        `REPERTOIRE_META` schema (one row per repertoire_id present in that locus)."""
        base = pl.DataFrame({
            "repertoire_id": list(repertoires.keys()),
            "source_files": [[f.name for f in files] for files in repertoires.values()],
            "patient_id" : [repertoires_to_patients[rep_id] for rep_id in repertoires.keys()]
        })

        for locus in sorted(present_loci):   # present_loci already excludes _unassigned
            rep_sizes = [
                pl.scan_parquet(f).select(
                    pl.first("repertoire_id"),
                    pl.sum("filter_pass").alias("n_clonotypes"),
                    pl.len().alias("n_filtered_clonotypes"),
                    pl.sum('duplicate_count').alias("total_duplicates")
                ).with_columns(
                    pl.col("n_filtered_clonotypes") - pl.col("n_clonotypes")
                )
                for f in self.db_dir.glob(loci_glob(locus))
            ]
            if not rep_sizes:
                continue
            rep_sizes = pl.concat(rep_sizes).collect(engine="streaming")

            # inner join: a repertoire only appears in a locus's meta if it produced that locus.
            df = base.join(rep_sizes, on="repertoire_id", how="inner").with_columns(
                pl.col("n_clonotypes").fill_null(0),
                pl.col("total_duplicates").fill_null(0)
            ).select(REPERTOIRE_META.keys()).cast(REPERTOIRE_META)

            df.write_parquet(self.db_dir / repertoire_meta_relpath(locus))

    def _generate_patient_metadata(self, repertoires_to_patients):

        repertoires_by_patient = defaultdict(list)
        for rep_id, patient_id in repertoires_to_patients.items():
            repertoires_by_patient[patient_id].append(rep_id)
        
        df = pl.DataFrame({
            "patient_id": list(repertoires_by_patient.keys()),
            "patient_repertoires": list(repertoires_by_patient.values()),
            "n_repertoires": [len(repertoires) for repertoires in repertoires_by_patient.values()]
        })

        return df.select(PATIENT_META.keys()).cast(PATIENT_META)
    

    def _generate_publication_metadata(self):
        df = pl.DataFrame({
            "publication_id": self.publication_ids
        })

        return df.select(PUBLICATION_META.keys()).cast(PUBLICATION_META)
    
    def _generate_generation_metadata(self, data_dir: Path):
        df = pl.DataFrame({
            "dataset_name" : [self.db_name],
            "source": [str(data_dir)],
            "reader" : [self.reader.__repr__()],
            "repertoire_mapper" : [self.repertoire_mapper.__repr__()],
            "patient_mapper" : [self.patient_mapper.__repr__()],
        }).with_columns(
            pl.lit(datetime.now().date()).alias("created_on")
        )
        return df.select(GENERATION_META.keys()).cast(GENERATION_META)
    
    def _delete_existing_data(self):
        # Re-ingesting the source data makes every derived operation output stale (its inputs
        # may have changed), so clear operations/ alongside processed_repertoires. Both are
        # recreated empty by _create_structure; ops recompute on demand into the fresh dir.
        shutil.rmtree(self.db_dir / Layout.processed_dir.path)
        ops_dir = self.db_dir / Layout.operations_dir.path
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