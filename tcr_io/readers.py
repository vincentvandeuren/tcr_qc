import polars as pl
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Dict, ClassVar, Optional, Iterable, Literal
import gzip
from .expressions import to_imgt, trim_junction_to_cdr3, is_functional_tcr
from .filters import Filterer
from .grouper import Grouper
import csv


class BaseReader(ABC):
    """
    Repertoire reader steps: 
    When duplicate_group_when is 'by_run' (default):

    1. Read
    2. Process (standardize columns, map gene aliases to IMGT, trim junctions)
    3. Concatenate (if multiple files) 
    4. Group duplicates
    5. Filter

    When duplicate_group_when is 'by_file', step 3 and 4 are swapped
    When duplicate_group is None, step 4 is skipped
    """
    name: str = "abstract_reader"
    col_map: ClassVar[Dict[Iterable[str]|str, str]] = {} # Mapping from input to output
    null_values: ClassVar[List[str]] = [] # Values to treat as null
    # Per-reader default grouper. Subclasses can point this at a different
    # Grouper (e.g. a future SingleCellGrouper) or set it to None to skip
    # grouping. An explicit `grouper=` arg to __init__ overrides it.
    default_grouper: ClassVar[Grouper | None] = Grouper()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, col_map={self.col_map})"

    def __init__(self,
                 filterer: Filterer | None = None,
                 grouper: Grouper | None = None,
                 duplicate_group_when : Optional[Literal["by_file", "by_run"]] = "by_run",
                 ):
        self.filterer = Filterer() if filterer is None else filterer
        self.grouper = self.default_grouper if grouper is None else grouper
        self.duplicate_group_when = duplicate_group_when

    @property
    def _should_group(self) -> bool:
        return self.grouper is not None and self.duplicate_group_when in ("by_file", "by_run")

    def run(self, paths:str|Path|List[Path]) -> pl.LazyFrame:
        """
        Main method to run the reader:
        - Read and concatenate input files
        - Process the dataframe to standardize columns and perform transformations
        """
        if isinstance(paths, str):
            paths = Path(paths)

        if not isinstance(paths, List):
            if paths.is_file():
                return self._run_single(paths) # read single file
            else:
                paths = list(paths.parent.glob(paths.name)) # convert glob pattern to list of files
                return self._run_multiple(paths) # read multiple files
        else:
            return self._run_multiple(paths)
        
    def _run_single(self, file:Path) -> pl.LazyFrame:
        df = self._read(file)
        df = self._process(df)
        if self._should_group:
            df = self.grouper.run(df)
        df = self._filter(df)
        return df

    def _run_multiple(self, files:List[Path]) -> pl.LazyFrame:
        if self._should_group and self.duplicate_group_when == "by_file":
            dfs = pl.concat([
                self.grouper.run(self._process(self._read(file))) for file in files
            ])
        else:
            dfs = pl.concat([
                self._process(self._read(file)) for file in files
            ])
            if self._should_group and self.duplicate_group_when == "by_run":
                dfs = self.grouper.run(dfs)
        dfs = self._filter(dfs)
        return dfs

    def _read_header(self, f: Path) -> List[str]:
        if f.suffix == ".parquet":
            return pl.read_parquet(f, n_rows=0).columns, ""

        elif f.suffix == ".gz":
            with gzip.open(f, "rt") as file:
                for line in file:
                    header = line.strip()
                    if not header.startswith("#"):
                        break
        else:
            with open(f, "r") as file:
                for line in file:
                    header = line.strip()
                    if not header.startswith("#"):
                        break

        dialect = csv.Sniffer().sniff(header, delimiters="\t,")
        fields = next(csv.reader([header], dialect))
        sep = dialect.delimiter

        return fields, sep

    def _determine_col_map(self, header:List[str]) -> Optional[Dict[str, str]]:
        """
        If the column mapping is can be determined, return it.
        If the header does not match the expected columns, return None.
        """
        col_map = {}
        for input_cols, output_col in self.col_map.items():
            if isinstance(input_cols, str):
                input_cols = [input_cols]
            for col in input_cols:
                if col in header:
                    col_map[col] = output_col
                    break
            else:
                # return ValueError(f"None of the columns {input_cols} found in input file header: {header}")
                return None
        return col_map

    def _read(self, path:Path) -> pl.LazyFrame:
        """
        Read the input file as a Polars LazyFrame, handiling columns, formats, and compression
        """
        suffixes = set(path.suffixes)
        header, separator = self._read_header(path)
        col_map = self._determine_col_map(header)
        if col_map is None:
            raise ValueError(f"File: {path.name} Input file header does not match expected columns for {self.name} reader: {header}")

        if ".parquet" in suffixes:
            df = pl.scan_parquet(path, include_file_paths="file")

        else:
            df = pl.scan_csv(
                path,
                separator=separator,
                null_values=self.null_values,
                comment_prefix="#",
                include_file_paths="file"
            )
        
        df = (
            df
            .with_columns(pl.col("file").str.extract(r"([^/]+)$"))
            .select(list(col_map.keys())+["file"])
            .rename(col_map)
        )
        return df
    
    def _process(self, df:pl.LazyFrame) -> pl.LazyFrame:
        """
        Standardize the input dataframe:
        - Columns: junction_aa, junction, v_call, j_call, duplicate_count
        - Handle any necessary transformations:
            - Map gene aliases to imgt notation
            - Trim nucleotide sequences to CDR3 region
        - Keep any failed rows for QC purposes
        """
        pass

    def _filter(self, df:pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply any necessary filters to the dataframe, such as removing non-functional TCRs or singlets
        """
        if self.filterer is not None:
            df = self.filterer.run(df)
        return df
    


class MixcrReader(BaseReader):
    name = "mixcr"
    col_map = {
        "nSeqCDR3": "junction",
        "aaSeqCDR3": "junction_aa",
        ("allVHitsWithScore", "bestVGene"): "v_call",
        ("allJHitsWithScore", "bestJGene"): "j_call",
        ("uniqueMoleculeCount", "cloneCount", "readCount"): "duplicate_count",
    }
    
    def _process(self, df):
        df = df.with_columns(
            pl.col("v_call").str.extract(r"^([^(]+)", 1).str.replace(r"(^|[^/\\])[\\/]?DV", "${1}/DV"),
            pl.col("j_call").str.extract(r"^([^(]+)", 1).str.replace(r"(^|[^/\\])[\\/]?DV", "${1}/DV"), # remove score info and take first call if multiple
        ).with_columns(
            to_imgt(pl.col("v_call")),
            to_imgt(pl.col("j_call")),
            trim_junction_to_cdr3(pl.col("junction"), pl.col("junction_aa")),
        )
        return df


class AdaptiveReader(BaseReader):
    name = "adaptive"
    col_map = {
        ("nucleotide","rearrangement", "cdr3_rearrangement"): "junction",
        ("aminoAcid","amino_acid", "cdr3_amino_acid"): "junction_aa",
        ("vMaxResolved", "v_resolved"): "v_call",
        ("jMaxResolved", "j_resolved"): "j_call",
        ("count (templates/reads)","templates", "seq_reads", "copy", "count", "count (reads)"):"duplicate_count",
    }
    null_values = ["unknown", "unresolved", "NA", "na"]
    
    def _process(self, df):
        df = df.with_columns(
            to_imgt(pl.col("v_call")),
            to_imgt(pl.col("j_call")),
            trim_junction_to_cdr3(pl.col("junction"), pl.col("junction_aa")),
        )
        return df

    
class AirrReader(BaseReader):
    name = "airr"
    col_map = {
    "junction":"junction",
    "junction_aa":"junction_aa",
    "v_call":"v_call",
    "j_call":"j_call",
    ("umi_count","duplicate_count", "count"):"duplicate_count",
    }

    def _process(self, df):
        df =  df.with_columns(
            to_imgt(pl.col("v_call"), split_on_character=","), # airr format can have multiple calls separated by comma, take the first one
            to_imgt(pl.col("j_call"), split_on_character=","),
            trim_junction_to_cdr3(pl.col("junction"), pl.col("junction_aa")),
        )
        return df

        

class TcrdistReader(BaseReader):
    name = "tcrdist"
    col_map = {
        "cdr3":"junction_aa",
        "cdr3_nucseq":"junction",
        "v_gene":"v_call",
        "j_gene":"j_call",
        "clone_size":"duplicate_count"
    }
    
    def _process(self, df):
        df =  df.with_columns(
            to_imgt(pl.col("v_call")),
            to_imgt(pl.col("j_call")),
            trim_junction_to_cdr3(pl.col("junction"), pl.col("junction_aa")),
        )
        return df

        
class VlasovaReader(BaseReader):
    name = "vlasova"
    col_map = {
        "cdr3nt":"junction",
        "cdr3aa":"junction_aa",
        "v":"v_call",
        "j":"j_call",
        "count":"duplicate_count",
    }

    def _process(self, df):
        df =  df.with_columns(
            to_imgt(pl.col("v_call")),
            to_imgt(pl.col("j_call")),
            trim_junction_to_cdr3(pl.col("junction"), pl.col("junction_aa")),
        )
        return df

class CellrangerReader(BaseReader):
    name = "cellranger"
    # TODO(single-cell): replace with a SingleCellGrouper that dedups within a
    # cell while retaining `cell_id`. Until then, skip grouping so cells are not
    # collapsed into clonotypes.
    default_grouper = None
    col_map = {
        "cdr3_nt":"junction",
        "cdr3":"junction_aa",
        "v_gene":"v_call",
        "j_gene":"j_call",
        "umis":"duplicate_count",
        "is_cell":"is_cell",
        "barcode":"cell_id",
    }
    
    def _process(self, df):
        df =  df.with_columns(
            to_imgt(pl.col("v_call")),
            to_imgt(pl.col("j_call")),
            trim_junction_to_cdr3(pl.col("junction"), pl.col("junction_aa")),
        )
        return df
    
class SynapseReader(BaseReader):
    name = "synapse"
    col_map = {
        "cdr3":"junction",
        "cdr3_aa":"junction_aa",
        "v_call":"v_call",
        "j_call":"j_call",
        'fwr3_aa':'fwr3_aa',
        'fwr4_aa':'fwr4_aa',
        'fwr3':'fwr3',
        'fwr4':'fwr4',
        }
    def _process(self, df):
        df= df.with_columns(
            junction_aa = pl.col("fwr3_aa").str.slice(-1) + pl.col("junction_aa") + pl.col("fwr4_aa").str.slice(0,1),
            junction = pl.col("fwr3").str.slice(-3)+pl.col("junction")+pl.col("fwr4").str.slice(0,3),
            duplicate_count = pl.lit(1)
        ).with_columns(
            to_imgt(pl.col("v_call")),
            to_imgt(pl.col("j_call")),
            trim_junction_to_cdr3(pl.col("junction"), pl.col("junction_aa")),
        )
        return df
    
class PirdReader(AirrReader, BaseReader):
    name = "pird"
    col_map = {
    "ntCDR3":"junction",
    "aaCDR3":"junction_aa",
    "vGene":"v_call",
    "jGene":"j_call",
    ("seqCount"):"duplicate_count",
    }
    null_values = ["na"]


class BradleyReader(AirrReader, BaseReader):
    name = "tcrdist_without_clone_count_for_nicaragua_data"
    col_map = {
    "cdr3_nucseq":"junction",
    "cdr3":"junction_aa",
    "v_gene":"v_call",
    "j_gene":"j_call",
    }
    
    def _process(self, df):
        df = df.with_columns(
            duplicate_count = pl.lit(1)
        )
        return super()._process(df)

class RosatiReader(AirrReader, BaseReader):
    name = "rosati"
    col_map = {
        "nSeqCDR3": "junction",
        "aaVJ":"aaVJ",
        "cloneCount":"duplicate_count"
    }

    def _process(self, df: pl.DataFrame) -> pl.DataFrame:
        aavj_pat = r"(?P<junction_aa>[A-Z]+)_(?P<v_call>TR[A-Z0-9-]+)_(?P<j_call>TR[A-Z0-9-]+)"
        df = df.with_columns(
            pl.col("aaVJ").str.extract_groups(aavj_pat)
        ).unnest("aaVJ")
        return super()._process(df)

class PogorelyyMixcrReader(MixcrReader, BaseReader):
    name = "mixcr_from_pogorelyy_2018"
    col_map = {
        "N. Seq. CDR3":"junction",
        "AA. Seq. CDR3":"junction_aa",
        "All V hits":"v_call",
        "All J hits":"j_call",
        "Clone count":"duplicate_count",
    }

class KoshlanTcrdistReader(AirrReader, BaseReader):
    name = "tcrdist3_koshlan"
    col_map = {
        "cdr3_b_aa":"junction_aa",
        "cdr3_rearrangement":"junction",
        "v_b_gene":"v_call",
        "j_b_gene":"j_call",
        "templates":"duplicate_count"
    }

immunarch_gene_map = {
    'TRBV13-1': 'TRBV13',
    'TRBV14-1': 'TRBV14',
    'TRBV15-1': 'TRBV15',
    'TRBV16-1': 'TRBV16',
    'TRBV18-1': 'TRBV18',
    'TRBV19-1': 'TRBV19',
    'TRBV2-1': 'TRBV2',
    'TRBV27-1': 'TRBV27',
    'TRBV28-1': 'TRBV28',
    'TRBV30-1': 'TRBV30',
    'TRBV9-1': 'TRBV9',
    'TRBV6-2/06-3': 'TRBV6-2',
    'TRBV12-3/12-4': 'TRBV12-3',
    'TRBV3-1/03-2': 'TRBV3-1',
}

class ImmunArchReader(AirrReader, BaseReader):
    name = "immunearch"
    col_map = {
    "CDR3.nt":"junction",
    "CDR3.aa":"junction_aa",
    "V.name":"v_call",
    "J.name":"j_call",
    "Clones":"duplicate_count",
    }

    def _process(self, df):
        df = df.with_columns(
            pl.col("v_call").replace(immunarch_gene_map)
        )
        return super()._process(df)
    
class TcrDbReader(AirrReader, BaseReader):
    name = "tcrdb"
    col_map = {
        "NNSeq":"junction",
        "AASeq":"junction_aa",
        "Vregion":"v_call",
        "Jregion":"j_call",
        ("cloneCount"):"duplicate_count",
    }

class ReaderFactory:
    base_reader: ClassVar[BaseReader] = BaseReader # for accessing shared methods like _determine_col_map

    def __init__(self, readers:Optional[List[BaseReader]]=None):
        if readers is not None:
            self.readers = readers
        else:
            self.readers = [cls() for cls in BaseReader.__subclasses__()]
        
        self.last_resolved_reader_ = None

    
    def run(self, paths:str|Path|List[Path]) -> pl.LazyFrame:

        if isinstance(paths, str):
            paths = Path(paths)
        if isinstance(paths, Path):
            if paths.is_file():
                paths = [paths]
            elif "*" in paths.name:
                paths = list(paths.parent.glob(paths.name))
        if not isinstance(paths, list):
            paths = list(paths)  # materialize generators / other iterables so paths[0] works

        first_file = paths[0]
        header, sep = self.base_reader()._read_header(first_file)

        for reader in self.readers:
            if reader._determine_col_map(header) is not None:
                self.last_resolved_reader_ = reader
                return reader.run(paths)
        
        raise ValueError(f"File: {first_file.name} Input file header does not match expected columns for any supported reader: {header}")
    
    def __repr__(self):
        return f"ReaderFactory, last_resolved_reader={self.last_resolved_reader_}"

