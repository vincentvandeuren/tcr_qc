"""
build_resources.py — generate the bundled model parquets for `tcr_io/resources/`.

This is the single, reproducible converter for the closed-beta packaging effort.
It turns the raw upstream model artifacts into the pre-processed parquet files the
package will ship (see docs/superpowers/specs/2026-06-30-pip-installable-package-design.md §6).

Inputs (in the repo-root `model_sources/` folder; gitignored, obtained separately):
  - tcr2hla_info.zip            : TCR2HLA model archive.
  - CMV_ECOcluster_TCRs.tsv     : CMV EcoCluster TCR table (raw TSV).
  - mait_parsed.parquet         : MAIT reference TCRs (already IMGT-processed).

Outputs (written to tcr_io/resources/ in --build mode, or scratch in --inspect mode):
  - hla_tcrs.parquet           : junction_aa, v_gene, j_gene, allele, gene, score
  - hla_model_weights.parquet  : allele, coef_1, coef_2, intercept
  - cmv_ecocluster.parquet     : v_gene, j_gene, junction_aa, hla_cocluster, eco_id
  - mait_hits.parquet          : v_gene, j_gene, junction_aa

The HLA/CMV transforms reproduce the exact logic currently in
`tcr_io/operations/hla.py` and `tcr_io/operations/cmv_hits.py`, so the bundled
parquets are drop-in replacements for the old pickle/TSV loading. Gene-name
canonicalisation uses the package's Rust `to_imgt` expression; that requires the
compiled extension (`maturin develop`). `--inspect` skips it to size the output.

Usage (from the repo root):
  python scripts/build_resources.py --inspect   # measure sizes, no to_imgt, scratch output
  python scripts/build_resources.py --build      # final parquets into tcr_io/resources/
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import sys
import zipfile
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# make the in-tree `tcr_io` package importable regardless of where this is run from
sys.path.insert(0, str(REPO_ROOT))

SOURCES_DIR = REPO_ROOT / "model_sources"
RESOURCES_DIR = REPO_ROOT / "tcr_io" / "resources"

HLA_ZIP = SOURCES_DIR / "tcr2hla_info.zip"
CMV_TSV = SOURCES_DIR / "CMV_ECOcluster_TCRs.tsv"
MAIT_PARQUET = SOURCES_DIR / "mait_parsed.parquet"

# Entries we need out of the (possibly truncated) zip.
HLA_WEIGHTS_ENTRY = "tcr2hla_info/models/TRB_weights.pickle.gz"
HLA_MODELS_ENTRY = "tcr2hla_info/models/TRB_models.pickle"

V_GENE_RE = r"(.*)\*\d{2}"  # strip the *NN allele suffix -> bare gene

# --------------------------------------------------------------------------- #
# Parquet compression: why zstd-19
# --------------------------------------------------------------------------- #
# These parquets are WRITE-ONCE (here, at package-build time) and READ-MANY
# (every HLA/CMV inference run, by every installed user). That profile means we
# only care about two things: final size (it ships inside the wheel) and read /
# decompression speed (user experience). Compression *speed* is irrelevant — we
# pay it once — so codecs that trade ratio for fast writes (snappy, lz4) just
# waste wheel space for us.
#
# How the codecs work:
#   - lz4 / snappy : LZ77 dictionary matching (replace repeated byte runs with
#                    back-references) with little/no entropy coding. Built for
#                    raw speed; weakest ratio.
#   - gzip/DEFLATE : LZ77 + Huffman entropy coding. Good ratio, but 1990s-era
#                    and slower to read than zstd.
#   - brotli       : LZ77 + Huffman + a built-in static dictionary. Great for
#                    web text, but here it's both bigger AND ~3-4x slower to read.
#   - zstd         : LZ77 + a modern entropy stage (FSE/tANS + Huffman). The
#                    `level` only tunes how hard the writer searches for matches
#                    (higher = smaller + slower WRITE); decompression speed is
#                    roughly constant across levels. So at build time we can crank
#                    the level for free and still get fast reads.
#
# Measured on the 1,049,766-row HLA table (this machine):
#   codec             size(MB)   write(s)   read(s)
#   uncompressed        27.45      0.14      0.069
#   lz4                 16.37      0.16      0.029
#   snappy              16.02      0.20      0.045
#   gzip                12.81      0.64      0.083
#   zstd-3 (default)    13.34      0.21      0.048
#   zstd-9              12.44      0.57      0.059
#   zstd-19             12.00      7.67      0.065   <-- chosen
#   zstd-22             12.00     12.12      0.066   (no gain over 19)
#   brotli              13.57      0.34      0.205
#
# zstd-19 is the smallest (12.00 MB) with reads as fast as the speed-focused
# codecs; its ~8s write is a one-time build cost we don't care about. The codec
# is recorded in the parquet footer, so any compliant reader auto-detects it —
# `pl.read_parquet(path)` needs no matching setting and the filename carries no
# codec suffix.
PARQUET_COMPRESSION = "zstd"
PARQUET_LEVEL = 19


def write_resource(df: pl.DataFrame, path: Path) -> None:
    df.write_parquet(path, compression=PARQUET_COMPRESSION, compression_level=PARQUET_LEVEL)


# --------------------------------------------------------------------------- #
# Step 0: read the two needed entries out of the zip
# --------------------------------------------------------------------------- #
def read_zip_entries(zip_path: Path, wanted: set[str]) -> dict[str, bytes]:
    """Return {entry_name: raw_bytes} for the wanted entries."""
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        missing = wanted - names
        if missing:
            raise ValueError(f"entries not found in {zip_path.name}: {sorted(missing)}")
        return {nm: z.read(nm) for nm in wanted}


# --------------------------------------------------------------------------- #
# to_imgt: package Rust expression if available; else identity (inspect only)
# --------------------------------------------------------------------------- #
def get_to_imgt(require: bool):
    try:
        from tcr_io.expressions import to_imgt  # needs the compiled extension

        return to_imgt, True
    except Exception as e:  # noqa: BLE001
        if require:
            raise SystemExit(
                "to_imgt unavailable (build the extension: `maturin develop`). "
                f"Original import error: {e!r}"
            )
        return None, False


# --------------------------------------------------------------------------- #
# Step 1: HLA scored-TCR table  (mirrors hla.py:_prepare_tcr_df)
# --------------------------------------------------------------------------- #
def build_hla_tcrs(weights_gz: bytes, to_imgt, apply_imgt: bool) -> pl.DataFrame:
    weights = pickle.loads(gzip.decompress(weights_gz))  # {hla: [(tcr, score), ...]}

    frames = []
    for hla, wts in weights.items():
        tcrs, scores = zip(*wts)
        frames.append(
            pl.DataFrame({"tcr": tcrs, "score": scores}).with_columns(
                pl.lit(hla).alias("allele"),
                pl.lit(hla.split("-")[0]).alias("gene"),
            )
        )

    df = pl.concat(frames).with_columns(
        junction_aa=pl.col("tcr").str.split("+").list.get(0),
        v_call=pl.col("tcr").str.split("+").list.get(1),
        j_call=pl.col("tcr").str.split("+").list.get(2),
    )

    if apply_imgt:
        df = df.with_columns(to_imgt(pl.col("v_call")), to_imgt(pl.col("j_call")))

    df = df.with_columns(
        pl.col("v_call").str.extract(V_GENE_RE).alias("v_gene"),
        pl.col("j_call").str.extract(V_GENE_RE).alias("j_gene"),
    ).select(["junction_aa", "v_gene", "j_gene", "allele", "gene", "score"])
    return df


# --------------------------------------------------------------------------- #
# Step 2: HLA model coefficients  (mirrors hla.py:_prepare_model_wts)
# --------------------------------------------------------------------------- #
def build_hla_model_weights(models_pickle: bytes) -> pl.DataFrame:
    models = pickle.loads(models_pickle)  # {hla: sklearn LogisticRegression}
    rows = [[hla, m.coef_[0, 0], m.coef_[0, 1], m.intercept_[0]] for hla, m in models.items()]
    return pl.DataFrame(rows, schema=["allele", "coef_1", "coef_2", "intercept"], orient="row")


# --------------------------------------------------------------------------- #
# Step 3: CMV EcoCluster table  (mirrors cmv_hits.py:_prepare_eco_df)
# --------------------------------------------------------------------------- #
def build_cmv(tsv_path: Path, to_imgt, apply_imgt: bool) -> pl.DataFrame:
    df = pl.read_csv(tsv_path, separator="\t").with_columns(
        pl.col("tcr")
        .str.extract_groups(r"(?P<junction_aa>[A-Z]+)\+(?P<v_call>[A-Z0-9-]+)\+(?P<j_call>[A-Z0-9-]+)")
        .struct.unnest()
    ).drop("tcr")

    if apply_imgt:
        df = df.with_columns(to_imgt("v_call"), to_imgt("j_call"))

    df = df.with_columns(
        pl.col("v_call").str.extract(V_GENE_RE).alias("v_gene"),
        pl.col("j_call").str.extract(V_GENE_RE).alias("j_gene"),
    ).with_row_index("eco_id").select(
        ["v_gene", "j_gene", "junction_aa", "hla_cocluster", "eco_id"]
    )
    return df


def build_mait(parquet_path: Path) -> pl.DataFrame:
    # Source is already IMGT-processed (has v_gene/j_gene); keep just the join
    # columns MaitHits uses, deduped and null-free. No to_imgt needed.
    return (
        pl.read_parquet(parquet_path)
        .select(["v_gene", "j_gene", "junction_aa"])
        .drop_nulls()
        .unique()
    )


def _report(name: str, df: pl.DataFrame, path: Path) -> None:
    size = path.stat().st_size
    print(f"  {name:<22} rows={df.height:>10,}  cols={df.width}  ->  {size/1e6:6.2f} MB  ({path})")


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true", help="measure sizes; skip to_imgt; scratch output")
    mode.add_argument("--build", action="store_true", help="write final parquets to tcr_io/resources/")
    ap.add_argument("--out", type=Path, default=None, help="override output dir (inspect mode)")
    args = ap.parse_args()

    apply_imgt = args.build
    to_imgt, have_imgt = get_to_imgt(require=args.build)
    out_dir = args.out or (RESOURCES_DIR if args.build else SOURCES_DIR / "_inspect_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"mode={'build' if args.build else 'inspect'}  to_imgt_applied={apply_imgt}  out={out_dir}\n")

    print("reading HLA model entries from zip ...")
    entries = read_zip_entries(HLA_ZIP, {HLA_WEIGHTS_ENTRY, HLA_MODELS_ENTRY})

    print("building parquets ...")
    hla_tcrs = build_hla_tcrs(entries[HLA_WEIGHTS_ENTRY], to_imgt, apply_imgt)
    hla_wts = build_hla_model_weights(entries[HLA_MODELS_ENTRY])
    cmv = build_cmv(CMV_TSV, to_imgt, apply_imgt)
    mait = build_mait(MAIT_PARQUET)

    p1 = out_dir / "hla_tcrs.parquet"
    p2 = out_dir / "hla_model_weights.parquet"
    p3 = out_dir / "cmv_ecocluster.parquet"
    p4 = out_dir / "mait_hits.parquet"
    write_resource(hla_tcrs, p1)
    write_resource(hla_wts, p2)
    write_resource(cmv, p3)
    write_resource(mait, p4)

    print("\nresults:")
    _report("hla_tcrs", hla_tcrs, p1)
    _report("hla_model_weights", hla_wts, p2)
    _report("cmv_ecocluster", cmv, p3)
    _report("mait_hits", mait, p4)
    total = p1.stat().st_size + p2.stat().st_size + p3.stat().st_size + p4.stat().st_size
    print(f"\n  TOTAL bundled size: {total/1e6:.2f} MB")
    if not apply_imgt:
        print("  NOTE: --inspect ran WITHOUT to_imgt; gene strings are not IMGT-canonicalised.")


if __name__ == "__main__":
    main()
