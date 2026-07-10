import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.special import gammaln
from .base import ALL_LOCI, BaseOperation, OperationResults


def get_fixed_depth_grid(num_points: int = 200, max_depth: int = 10_000_000) -> np.ndarray:
    """
    Fixed log-spaced grid with nice round numbers.
    Powers of 10 always included. All values rounded to 2 significant figures.
    """
    max_exp = int(np.floor(np.log10(max_depth)))

    # Anchor: all powers of 10
    powers_of_10 = np.array([10**i for i in range(max_exp + 1)])

    # Fill: log-spaced, rounded to 2 sig figs
    raw = np.logspace(0, np.log10(max_depth), num=num_points)
    rounded = _round_sig_figs(raw, sig=2)

    grid = np.unique(np.concatenate([powers_of_10, rounded]))
    return grid


def _round_sig_figs(arr: np.ndarray, sig: int = 2) -> np.ndarray:
    """Round array values to N significant figures."""
    arr = np.asarray(arr, dtype=float)
    arr = np.maximum(arr, 1)
    exp = np.floor(np.log10(arr))
    factor = 10 ** (exp - sig + 1)
    return (np.round(arr / factor) * factor).astype(int)

def compute_frequency_spectrum(lf: pl.LazyFrame) -> tuple[np.ndarray, np.ndarray]:
    freq = (
        lf
        .group_by("duplicate_count")
        .agg(pl.len().alias("species_count"))
        .sort("duplicate_count")
        .collect()
    )
    return (
        freq["duplicate_count"].to_numpy().astype(np.int64),
        freq["species_count"].to_numpy().astype(np.int64),
    )


def compute_sample_stats(lf: pl.LazyFrame) -> dict:
    stats = (
        lf
        .select(
            pl.len().alias("s_obs"),
            pl.col("duplicate_count").sum().alias("total_abundance"),
            (pl.col("duplicate_count") == 1).sum().alias("f1"),
            (pl.col("duplicate_count") == 2).sum().alias("f2"),
        )
        .collect()
        .row(0, named=True)
    )
    return {k: int(v) for k, v in stats.items()}


def rarefy(unique_ni: np.ndarray, ni_counts: np.ndarray,
           s_obs: int, N: int, depths: np.ndarray) -> np.ndarray:
    if len(depths) == 0:
        return np.array([])

    results = np.full(len(depths), float(s_obs))
    results[depths == 0] = 0.0

    mask = (depths > 0) & (depths < N)
    m = depths[mask].astype(np.float64)
    if len(m) == 0:
        return results

    m_col = m[:, np.newaxis]
    ni_row = unique_ni[np.newaxis, :].astype(np.float64)
    N_minus_ni = N - ni_row

    log_p = (
        gammaln(N_minus_ni + 1) -
        gammaln(N_minus_ni - m_col + 1) -
        gammaln(N + 1) +
        gammaln(N - m_col + 1)
    )

    p_miss = np.exp(log_p)
    p_miss[m_col > N_minus_ni] = 0.0

    sum_p_miss = (p_miss * ni_counts[np.newaxis, :]).sum(axis=1)
    results[mask] = s_obs - sum_p_miss

    return results


def extrapolate_polars(s_obs: int, f1: int, f2: int,
                       N: int, depths: np.ndarray) -> pl.DataFrame:
    if f2 > 0:
        f0_hat = ((N - 1) / N) * (f1 ** 2) / (2 * f2)
    else:
        f0_hat = ((N - 1) / N) * (f1 * (f1 - 1)) / 2

    depth_col = pl.col("subsampling_depth").cast(pl.Float64)

    if f1 == 0 or f0_hat == 0:
        return pl.DataFrame({"subsampling_depth": depths}).with_columns(
            pl.lit(float(s_obs)).alias("expected_richness"),
            pl.lit("Extrapolation").alias("type"),
        )

    rate = f1 / (N * f0_hat + f1)

    return pl.DataFrame({"subsampling_depth": depths}).with_columns(
        (
            pl.lit(float(s_obs)) +
            pl.lit(f0_hat) * (
                1.0 - pl.lit(1.0 - rate).pow(depth_col - pl.lit(float(N)))
            )
        ).alias("expected_richness"),
        pl.lit("Extrapolation").alias("type"),
    )


def compute_rarefaction_curve(
    lf: pl.LazyFrame,
    depth_grid: np.ndarray,
    extrapolation: bool = True,
) -> pl.DataFrame:
    stats = compute_sample_stats(lf)
    s_obs, N, f1, f2 = stats["s_obs"], stats["total_abundance"], stats["f1"], stats["f2"]

    if N == 0:
        return pl.DataFrame({
            "subsampling_depth": [0],
            "expected_richness": [0.0],
            "type": ["Interpolation"],
            "total_abundance": [0],
        })

    unique_ni, ni_counts = compute_frequency_spectrum(lf)

    interp_depths = depth_grid[depth_grid <= N]
    extrap_depths = depth_grid[depth_grid > N]

    interp_richness = rarefy(unique_ni, ni_counts, s_obs, N, interp_depths)

    interp_df = pl.DataFrame({
        "subsampling_depth": interp_depths,
        "expected_richness": interp_richness,
        "type": ["Interpolation"] * len(interp_depths),
    })

    if extrapolation and len(extrap_depths) > 0:
        extrap_df = extrapolate_polars(s_obs, f1, f2, N, extrap_depths)
        result = pl.concat([interp_df, extrap_df])
    else:
        result = interp_df

    return result.with_columns(pl.lit(N).alias("total_abundance"))


@dataclass
class RarefactionReport(BaseOperation):
    name = "rarefaction_report"
    version = "0.1"
    description = "Rarefaction/extrapolation curves per repertoire."
    supported_loci = ALL_LOCI

    num_points: int = 200
    max_depth: int = 10_000_000
    extrapolation: bool = True

    def _run(self, ds, locus: Optional[str] = None) -> OperationResults:
        # Compute grid once — shared by all samples
        depth_grid = get_fixed_depth_grid(num_points=self.num_points, max_depth=self.max_depth)

        result = ds.map_repertoires(
            lambda rep: compute_rarefaction_curve(
                rep, depth_grid=depth_grid, extrapolation=self.extrapolation,
            ),
            locus=locus, filter_pass_only=True, progress_bar=True,
            progress_desc="Computing rarefaction",
        )

        return OperationResults(outputs={
            "rarefaction_curves": result,
        })