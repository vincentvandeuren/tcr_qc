from polars import DataFrame

__version__: str

def hd1_pairs(
    df: DataFrame,
    sequence: str = ...,
    by: list[str] = ...,
    max_distance: int = ...,
    metric: str = ...,
) -> DataFrame: ...

def neighbor_counts(
    df: DataFrame,
    sequence: str = ...,
    by: list[str] = ...,
    max_distance: int = ...,
    metric: str = ...,
) -> DataFrame: ...
