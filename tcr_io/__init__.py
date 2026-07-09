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
from .grouper import Grouper
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
    "Grouper",
    "operations",
]
