"""Build the TRB CDR3 index of one dataset; with --migrate, then migrate it to v8 (drops its tabulation).

    python scripts/rebuild_indexes.py <dataset_dir> [--migrate]

Only TRB is handled: other loci are neither indexed nor looked at, and a failure there blocks nothing. The dataset
is migrated only if its TRB index record is `success`. On a build failure it stays at
v7 with its tabulation, the reason is printed, and the exit status is 1. A dataset already at v8 with
successful records of the current builder version is skipped, so a rollout loop can be re-run after an interruption.
"""
import argparse
import resource
import sys
import time
from pathlib import Path

from tcr_io import Dataset
from tcr_io.operations import ConstructBazairrIndex

INDEX_OP, TABULATION_OP = "construct_bazairr_index", "tabulate_by_vj_gene"
LOCUS = "TRB"   # the rollout policy: the operation itself supports more loci


def du(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def index_records(ds: Dataset) -> list:
    return [r for r in ds.operations if r.operation_name == INDEX_OP and r.locus == LOCUS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--migrate", action="store_true")
    args = ap.parse_args()
    path, name = args.dataset_dir, args.dataset_dir.name

    ds = Dataset(path)
    if LOCUS not in ds.dispatch_loci:
        print(f"{name}: no {LOCUS}, skipped")
        return 0
    ds = ds.select_locus(LOCUS)
    records = index_records(ds)
    if ds.version >= 8 and records and all(
            r.status == "success" and r.version == ConstructBazairrIndex.version for r in records):
        print(f"{name}: already at v{ds.version} with a successful index, skipped")
        return 0

    tab_gb = du(path / "operations" / TABULATION_OP)   # before migrating deletes it
    start = time.time()
    ds.run_operation(ConstructBazairrIndex())          # loci with a successful record are reused
    build_s = time.time() - start
    records = index_records(ds)
    failed = [r for r in records if r.status != "success"]
    if failed or not records:
        for r in failed:
            print(f"{name}: index build failed on {r.locus}: {r.error}", file=sys.stderr)
        if not records:
            print(f"{name}: no index record was written", file=sys.stderr)
        print(f"{name}: left at v{ds.version}, tabulation intact", file=sys.stderr)
        return 1

    if args.migrate:
        ds = Dataset.migrated(path).select_locus(LOCUS)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # KB on Linux
    print(f"{name}: build {build_s:.0f} s, peak RSS {rss_gb:.1f} GB, index {du(path / 'operations' / INDEX_OP):.1f} GB, "
          f"tabulation {tab_gb:.1f} GB{' (freed)' if args.migrate else ' (kept)'}, final v{ds.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
