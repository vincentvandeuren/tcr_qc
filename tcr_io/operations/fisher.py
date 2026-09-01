from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import polars as pl
from scipy.stats import fisher_exact

from .base import BaseOperation
from .tabulate import TabulateByVJ
from ..expressions import KNOWN_LOCI, extract_genes
from ..structure import Artifact, Format, Store


@lru_cache(maxsize=1_000_000)
def _fisher(a: int, b: int, c: int, d: int, alternative: str) -> float:
    """Fisher-exact p-value on the 2x2 table [[a, b], [c, d]]; nan on an invalid table.

    Cached because the p-value depends only on the four cell counts, so the caller evaluates
    it once per distinct (n_positive, n_negative) contingency shape (totals are constant)."""
    try:
        return float(fisher_exact([[a, b], [c, d]], alternative=alternative).pvalue)
    except ValueError:
        return float("nan")


_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass
class FisherTest:
    """One association test config. `name` is a safe identifier that becomes the `{test}`
    parameter of `FisherAssociation.association` — i.e. one
    ``operations/fisher_association/<locus>/<name>.parquet`` file per test.

    The binary label is derived from ``label_column`` (a column of ``full_patient_meta`` or
    ``full_repertoire_meta``): rows whose value is in ``positive`` -> True; rows in ``negative``
    -> False; ``negative=None`` means every other **non-null** value is False. Rows matching
    neither side are null and dropped."""
    name: str
    label_column: str
    positive: object = True
    negative: Optional[object] = None
    alternative: Literal["two-sided", "greater"] = "two-sided"

    def __post_init__(self):
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"FisherTest.name {self.name!r} must be a valid identifier "
                f"(letters, digits, underscore; not starting with a digit)."
            )

    def label_expr(self) -> pl.Expr:
        col = pl.col(self.label_column)
        pos = list(self.positive) if isinstance(self.positive, (list, tuple)) else [self.positive]
        if self.negative is None:
            return (pl.when(col.is_in(pos)).then(pl.lit(True))
                      .when(col.is_not_null()).then(pl.lit(False))
                      .otherwise(None)).alias("label")
        neg = list(self.negative) if isinstance(self.negative, (list, tuple)) else [self.negative]
        return (pl.when(col.is_in(pos)).then(pl.lit(True))
                  .when(col.is_in(neg)).then(pl.lit(False))
                  .otherwise(None)).alias("label")


@dataclass
class FisherAssociation(BaseOperation):
    name = "fisher_association"
    version = "0.1"
    description = "Emerson associated-TCR test: per-clonotype Fisher-exact association of clone presence with a binary phenotype label."
    supported_loci = KNOWN_LOCI

    # One table per configured test. Parameterised because the test list is config, not code:
    # the number of outputs is only known once the op is constructed, so it cannot be a fixed
    # set of class attributes. The runner records what was written rather than what was
    # declared, so zero tests is a legal (empty) result rather than a missing output.
    association = Artifact("{test}.parquet", Format.PARQUET)

    tests: Optional[List[FisherTest]] = None        # set in __post_init__ (mutable default guard)
    grain: Literal["patient", "repertoire"] = "patient"
    pseudocount: float = 1.0
    min_incidence: int = 2                          # drop clones with total incidence < this

    def __post_init__(self):
        if self.tests is None:
            self.tests = [FisherTest(name="cmv", label_column="cmv_status", positive=True)]
        # coerce dicts (e.g. from ds.rerun stored params) back into FisherTest
        self.tests = [t if isinstance(t, FisherTest) else FisherTest(**t) for t in self.tests]
        names = [t.name for t in self.tests]
        if len(set(names)) != len(names):
            raise ValueError(f"FisherTest names must be unique; got {names}.")

    def _associate(self, incidence: pl.DataFrame, total_positive: int,
                   total_negative: int, alternative: str) -> pl.DataFrame:
        p = self.pseudocount
        # Fisher ONCE per distinct (n_positive, n_negative) shape, then join back.
        shapes = (
            incidence.select("n_positive", "n_negative").unique()
            .with_columns(_c=total_positive - pl.col("n_positive"),
                          _d=total_negative - pl.col("n_negative"))
            .with_columns(p_value=pl.struct("n_positive", "n_negative", "_c", "_d").map_elements(
                lambda s: _fisher(int(s["n_positive"]), int(s["n_negative"]),
                                  int(s["_c"]), int(s["_d"]), alternative),
                return_dtype=pl.Float64))
            .select("n_positive", "n_negative", "p_value")
        )
        return (
            incidence.join(shapes, on=["n_positive", "n_negative"], how="left")
            .with_columns(
                total_positive=pl.lit(total_positive, dtype=pl.Int64),
                total_negative=pl.lit(total_negative, dtype=pl.Int64),
                lfc=((pl.col("n_positive") + p) / (total_positive + p)).log(base=2)
                    - ((pl.col("n_negative") + p) / (total_negative + p)).log(base=2),
            )
            .select("v_gene", "junction_aa", "j_gene", "n_positive", "n_negative",
                    "total_positive", "total_negative", "p_value", "lfc")
            .sort("p_value")
        )

    def _run(self, ds, out: Store) -> None:
        tab_dir = self._ensure_tabulated(ds)
        key = "patient_id" if self.grain == "patient" else "repertoire_id"

        meta = ds.full_patient_meta if self.grain == "patient" else ds.full_repertoire_meta
        missing = [t.label_column for t in self.tests if t.label_column not in meta.columns]
        if missing:
            raise ValueError(
                f"fisher_association: label column(s) {sorted(set(missing))} not in "
                f"full_{self.grain}_meta (available: {meta.columns})."
            )
        labels = {t.name: meta.select(key, t.label_expr()).drop_nulls("label") for t in self.tests}
        totals = {t.name: (int(labels[t.name]["label"].sum()),
                           labels[t.name].height - int(labels[t.name]["label"].sum()))
                  for t in self.tests}

        incidence = self._incidence(ds, tab_dir, key, labels)   # name -> incidence DataFrame

        for t in self.tests:
            tp, tn = totals[t.name]
            out(self.association, test=t.name).write(
                self._associate(incidence[t.name], tp, tn, t.alternative)
            )

    def _ensure_tabulated(self, ds) -> Path:
        """The TabulateByVJ output directory for this locus, running that op first if it is not
        there yet.

        The nested run goes through the runner like any other, so the directory is recorded
        against `tabulate_by_vj_gene` and cannot leak into this op's own outputs. `ds` is bound
        to one locus, and the runner fans out over the binding, so this costs one locus.""" 
        handle = ds.result(TabulateByVJ.tabulated)
        if not handle.exists():
            ds.run_operation(TabulateByVJ())
        return handle.path()

    def _incidence(self, ds, tab_dir: Path, key: str, labels: dict) -> dict:
        """Per-test per-clone (v_gene, j_gene, junction_aa) -> n_positive / n_negative, built one
        (v_gene, j_gene) shard at a time. A label-independent per-subject presence table is built
        once per shard and reused across every test (shared scan). Returns {test_name: DataFrame}."""
        rep2subject = (ds.repertoire_meta.select("repertoire_id", "patient_id")
                       if key == "patient_id" else None)   # already one row per repertoire
        parts = {name: [] for name in labels}
        for v_dir in sorted(tab_dir.iterdir()):
            for j_dir in sorted(v_dir.iterdir()):
                base = pl.scan_parquet(j_dir).with_columns(*extract_genes())  # re-derive v_gene/j_gene
                if rep2subject is not None:
                    base = base.join(rep2subject.lazy(), on="repertoire_id")
                presence = (base.select(key, "v_gene", "j_gene", "junction_aa")
                            .unique(subset=[key, "v_gene", "j_gene", "junction_aa"])
                            .collect(engine="streaming"))
                for name, lab in labels.items():
                    parts[name].append(
                        presence.join(lab, on=key, how="inner")
                        .group_by("v_gene", "j_gene", "junction_aa")
                        .agg(pl.col("label").sum().alias("n_positive"),
                             (~pl.col("label")).sum().alias("n_negative"))
                    )
        out = {}
        for name in labels:
            inc = pl.concat(parts[name]) if parts[name] else pl.DataFrame(
                schema={"v_gene": pl.Utf8, "j_gene": pl.Utf8, "junction_aa": pl.Utf8,
                        "n_positive": pl.UInt32, "n_negative": pl.UInt32})
            out[name] = inc.filter((pl.col("n_positive") + pl.col("n_negative")) >= self.min_incidence)
        return out
