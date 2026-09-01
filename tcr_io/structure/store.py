"""Paths and I/O: `Artifact`, `Handle`, `Store`.

Three types, one job each:

    Artifact   a well-known path SHAPE — a template, a format, and policies. Inert.
    Handle     one artifact + one root + one set of parameters. The ONLY object here
               that touches the filesystem.
    Store      a binder with no verbs: which root, which parameters are already known.
               Calling it with an Artifact hands back a Handle.

The split exists so that "where does this live", "what does it contain" and "how do I
read it" stop being three independent decisions made at each call site. It replaces
`writers.py` (a Kind-keyed write dispatch with no read counterpart) and the path-builder
half of `layout.py`.

An artifact whose template has an unbound parameter describes a *set* of files, and a
handle over it reads them as one frame — which is how a locus-partitioned table is read
across every locus without a bespoke glob helper.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from itertools import takewhile
from pathlib import Path
from string import Formatter
from typing import Callable, Iterable, Literal, Optional, Union

import polars as pl

log = logging.getLogger(__name__)


# --- codecs ---------------------------------------------------------------------------
#
# Every reader takes `schema=`; every writer takes `schema=`. The schema is not always
# used, but the uniform signature is what keeps `Handle` free of per-format branches.

def _read_parquet(p, schema: Optional[pl.Schema] = None) -> pl.DataFrame:
    return pl.read_parquet(p)


def _write_parquet(v, p: Path, schema: Optional[pl.Schema] = None) -> None:
    """Write a frame, streaming when it is lazy."""
    if isinstance(v, pl.LazyFrame):
        v.sink_parquet(p)
    else:
        v.write_parquet(p)


def _read_ndjson(p, schema: Optional[pl.Schema] = None) -> pl.DataFrame:
    """The schema is not optional in practice: `pl.read_ndjson` on a zero-row file raises
    ``ComputeError: Cannot infer NDJSON types on empty reader`` *during the read*, long
    before `Handle._cast` could fix the dtypes. Passing it through is the only thing that
    makes an empty ndjson file readable."""
    return pl.read_ndjson(p, schema=schema)


def _write_ndjson(v, p: Path, schema: Optional[pl.Schema] = None) -> None:
    """A frame goes out through polars; a list of records goes out a line at a time.

    No `default=str`. A value that will not serialise is an error here, where the code
    that produced it can still be fixed — not a string that reads back as the wrong type
    (see `operations.record.serialize_params`, which this rule exists for)."""
    if isinstance(v, pl.LazyFrame):
        v = v.collect()
    if isinstance(v, pl.DataFrame):
        v.write_ndjson(p)
        return
    with open(p, "w") as f:
        for record in v:
            f.write(json.dumps(record) + "\n")


def _read_json(p, schema: Optional[pl.Schema] = None):
    return json.loads(Path(p).read_text())


def _write_json(v, p: Path, schema: Optional[pl.Schema] = None) -> None:
    p.write_text(json.dumps(v, indent=2) + "\n")


class Format(Enum):
    """An extension and its codec in one object, so a member cannot exist without its I/O.

    `PARQUET_DIR` is a directory: `ext` is `""` and it has `scan` only. `read` and `write`
    are `None`, and reaching for them raises TypeError — a directory is sunk into and
    scanned back, never round-tripped through one frame. A `None` slot is a stated
    capability, not a gap.
    """
    #             ext          read           write           scan
    PARQUET     = (".parquet", _read_parquet, _write_parquet, pl.scan_parquet)
    NDJSON      = (".ndjson",  _read_ndjson,  _write_ndjson,  None)
    JSON        = (".json",    _read_json,    _write_json,    None)
    PARQUET_DIR = ("",         None,          None,           pl.scan_parquet)

    def __init__(self, ext: str, read: Optional[Callable], write: Optional[Callable],
                 scan: Optional[Callable]):
        self.ext, self.read, self.write, self.scan = ext, read, write, scan

    @property
    def is_dir(self) -> bool:
        """Named, because two checks branch on it and `ext == ""` reads like a coincidence."""
        return self is Format.PARQUET_DIR


class OnMissing(Enum):
    """What a read does when nothing matches. Declared per artifact, so no call site
    branches on absence."""
    RAISE = "raise"
    EMPTY = "empty"    # typed-empty frame, built from the schema; requires one
    NONE  = "none"     # return None; the caller decides


def _fields(template: str) -> list[str]:
    """The `{parameter}` names in a template, parsed with `str.format`'s own grammar."""
    return [name for _, name, _, _ in Formatter().parse(template) if name is not None]


def resolve(template: str, params: dict, *, mode: Literal["strict", "glob"] = "strict") -> str:
    """Substitute `{parameters}` into a template.

    strict  every parameter must be bound; a missing one is a KeyError
    glob    unbound parameters become `*`, giving a search pattern
    """
    if mode == "strict":
        missing = [f for f in _fields(template) if params.get(f) is None]
        if missing:
            raise KeyError(
                f"unbound parameter(s) {missing} for template {template!r}; "
                f"bound: {sorted(k for k, v in params.items() if v is not None)}"
            )
        return template.format(**params)
    if mode == "glob":
        return template.format(**{f: params.get(f) or "*" for f in _fields(template)})
    raise ValueError(f"unknown resolve mode {mode!r}")


@dataclass(frozen=True)
class Artifact:
    """One well-known path shape. Inert: it knows a template, a format, and its policies —
    it performs no I/O and holds no root.

    template    path relative to a Store's root; `{param}` placeholders only
    format      extension + codec
    schema      the frame IS this — exact; drives select+cast on read and on write
    requires    the frame CONTAINS this — a subset check, for user-supplied side tables
                whose join key must be present but whose other columns are free
    on_missing  RAISE / EMPTY / NONE
    writable    False for artifacts produced by a partitioned sink rather than a write
    key/owner   filled in by `__set_name__`

    `schema`/`requires`/`key`/`owner` are `compare=False`: a `pl.Schema` is unhashable, and a
    frozen dataclass that carries one cannot go in a set or a dict key. Equality therefore
    keys on the path shape and its policies — which is the identity that matters, since two
    artifacts at the same template with different schemas are a bug, not two objects.
    """
    template: str
    format: Format = Format.PARQUET
    schema: Optional[pl.Schema] = field(default=None, compare=False)
    requires: Optional[pl.Schema] = field(default=None, compare=False)
    on_missing: OnMissing = OnMissing.RAISE
    writable: bool = True
    key: str = field(default="", compare=False)
    owner: Optional[type] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for _, name, spec, conv in Formatter().parse(self.template):
            if name is None:
                continue
            if not name.isidentifier():
                raise ValueError(f"{self.template!r}: {name!r} is not a valid parameter name")
            if spec or conv:
                raise ValueError(f"{self.template!r}: format specs/conversions are not allowed")
        if self.format.is_dir:
            # Explicit, not accidental. The old check read `template.endswith(ext)`, and
            # every string ends with "", so a directory artifact passed a test it was never
            # subject to. A directory's real constraint is the opposite one.
            if Path(self.template).suffix:
                raise ValueError(f"{self.template!r}: a {self.format.name} artifact is a "
                                 f"directory and must not have a file extension")
        elif not Path(self.template).suffix:
            # A file artifact must be a file. The *exact* extension is not asserted here:
            # two legacy artifacts hold NDJSON under a `.json` name, and that mismatch is
            # data, not a modelling error — so it is named explicitly in `layout.py`
            # (`_WRONG_EXT`) and dies with the rename, rather than being tolerated silently
            # for every artifact forever.
            raise ValueError(f"{self.template!r}: a {self.format.name} artifact is a file "
                             f"and must have a file extension")
        if self.on_missing is OnMissing.EMPTY and self.schema is None:
            raise ValueError(f"{self.template!r}: on_missing=EMPTY needs a schema to build "
                             f"the empty frame from")

    def __set_name__(self, owner: type, name: str) -> None:
        object.__setattr__(self, "key", name)
        object.__setattr__(self, "owner", owner)

    @property
    def params(self) -> tuple[str, ...]:
        return tuple(_fields(self.template))


@dataclass(frozen=True)
class Handle:
    """One artifact + one root + one set of parameters. The only object that touches the
    filesystem.

    A handle with an unbound parameter is a *set* of files: `scan()` globs and reads them
    as one frame, `read()` refuses and says so.
    """
    art: Artifact
    root: Path
    params: dict = field(default_factory=dict)

    # --- paths ------------------------------------------------------------------------
    def path(self) -> Path:
        """Resolve strictly. Raises KeyError if a parameter is unbound."""
        return self.root / resolve(self.art.template, self.params, mode="strict")

    def glob(self) -> list[Path]:
        """Resolve in glob mode and list matches; a fully-bound template yields 0 or 1 path.

        Matches whatever the template describes — for a directory artifact that is
        DIRECTORIES, which is how a `locus=*` partition list is taken without a leaf-file
        glob that would make an empty partition vanish.
        """
        pattern = resolve(self.art.template, self.params, mode="glob")
        if "*" not in pattern:
            p = self.root / pattern
            return [p] if p.exists() else []
        return sorted(self.root.glob(pattern))

    def exists(self) -> bool:
        return bool(self.glob())

    # --- reads ------------------------------------------------------------------------
    def scan(self, **kw) -> Optional[pl.LazyFrame]:
        """Lazy read over every matching path. TypeError if the format is not scannable."""
        fmt = self.art.format
        if fmt.scan is None:
            raise TypeError(f"{fmt.name} is not scannable ({self.art.template!r}); use read()")
        paths = self.glob()
        if not paths:
            missing = self._missing()
            return missing.lazy() if isinstance(missing, pl.DataFrame) else missing
        return self._cast(fmt.scan(paths, **kw))

    def read(self, **kw):
        """Eager read of ONE path, cast to `schema`. Applies `on_missing` when nothing
        matches; raises when the handle matches several, because silently concatenating a
        set of files is `scan()`'s job and should be asked for by name."""
        fmt = self.art.format
        if fmt.read is None:
            raise TypeError(f"{fmt.name} has no reader ({self.art.template!r}); use scan()")
        paths = self.glob()
        if not paths:
            return self._missing()
        if len(paths) > 1:
            raise ValueError(
                f"{self.art.template!r} matches {len(paths)} paths with "
                f"{sorted(self.params)} bound; use scan() to read them as one frame"
            )
        return self._cast(fmt.read(paths[0], schema=self.art.schema, **kw))

    # --- writes -----------------------------------------------------------------------
    def write(self, value) -> Path:
        """Cast to `schema`, mkdir -p, write. TypeError if not writable, or if the format
        has no writer."""
        art = self.art
        if not art.writable:
            raise TypeError(f"{art.template!r} is not writable (it is produced by a sink)")
        if art.format.write is None:
            raise TypeError(f"{art.format.name} has no writer ({art.template!r})")
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        art.format.write(self._cast(value), p, schema=art.schema)
        return p

    def update(self, **fields) -> Path:
        """Read-modify-write for a JSON object: set `fields`, keep every other key —
        INCLUDING keys this library version does not know about.

        For the manifest, whose two writers hold it at different times: ingestion writes the
        whole value, and `Migrator` later sets `version` on a file it did not write and whose
        other keys may come from a NEWER library version than the one migrating. Merging is
        correct exactly there; it is a bug everywhere else. In particular NOT for an operation
        record, which is JSON but single-writer — merging those is how a failed rerun clobbers
        a prior run's output list. The format check cannot enforce that; the rule is "a writer
        that does not hold the whole value", not "is JSON".

        Works on the raw dict rather than a typed wrapper, so a key written by a NEWER
        library version survives a write from an older one.
        """
        if self.art.format is not Format.JSON:
            raise TypeError(f"update() is JSON-only ({self.art.template!r} is "
                            f"{self.art.format.name})")
        p = self.path()
        current = json.loads(p.read_text()) if p.exists() else {}
        if not isinstance(current, dict):
            raise TypeError(f"{p} holds a {type(current).__name__}, not a JSON object")
        current.update(fields)
        return self.write(current)

    # --- directories ------------------------------------------------------------------
    def dir(self) -> Path:
        """The deepest directory this artifact is known to live in — itself for a directory
        format, its parent otherwise, truncated at the first unbound parameter.

        Truncating rather than raising is what lets one function answer both "make the
        directories a dataset needs" and "does this tree look like a dataset": every
        artifact has a knowable home even when the files in it do not exist yet.
        """
        rel = Path(resolve(self.art.template, self.params, mode="glob"))
        parts = rel.parts if self.art.format.is_dir else rel.parent.parts
        return self.root.joinpath(*takewhile(lambda seg: "*" not in seg, parts))

    def clear(self) -> Path:
        """rmtree the directory and recreate it empty. Logs first. The only rmtree a
        Handle can reach."""
        if not self.art.format.is_dir:
            raise TypeError(f"clear() is for directory artifacts ({self.art.template!r} is "
                            f"{self.art.format.name})")
        p = self.path()
        log.info("clearing %s", p)
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # --- internals --------------------------------------------------------------------
    def _cast(self, v):
        """Enforce `requires`, then `select(schema).cast(schema)` when a schema is declared
        and `v` is a frame. Runs on the read path and the write path, so a side table
        missing its join key is rejected before it reaches disk."""
        if not isinstance(v, (pl.DataFrame, pl.LazyFrame)):
            return v
        self._check(v.collect_schema() if isinstance(v, pl.LazyFrame) else v.schema)
        if self.art.schema is None:
            return v
        return v.select(self.art.schema.keys()).cast(self.art.schema)

    def _check(self, schema: pl.Schema) -> None:
        """Every key in `requires` present, with the declared dtype."""
        req = self.art.requires
        if req is None:
            return
        missing = [c for c in req.keys() if c not in schema]
        if missing:
            raise ValueError(f"{self.art.template!r}: required column(s) {missing} absent "
                             f"(present: {list(schema)})")
        wrong = {c: (schema[c], dt) for c, dt in req.items() if schema[c] != dt}
        if wrong:
            raise ValueError(f"{self.art.template!r}: required column dtype mismatch: "
                             + ", ".join(f"{c} is {a}, need {b}" for c, (a, b) in wrong.items()))

    def _missing(self):
        """Apply `on_missing`."""
        art = self.art
        if art.on_missing is OnMissing.RAISE:
            raise FileNotFoundError(
                f"{resolve(art.template, self.params, mode='glob')} not found under {self.root}"
            )
        if art.on_missing is OnMissing.NONE:
            return None
        # EMPTY: build FROM the schema. An empty frame with inferred dtypes gives Null
        # columns, and a Null column joined to a String one raises SchemaError — which is
        # exactly the failure this branch exists to prevent.
        return pl.DataFrame(schema=art.schema)


@dataclass(frozen=True)
class Store:
    """A binder with no verbs: which root, which parameters are already known. Calling it
    with an artifact hands back a Handle.

    Frozen — every binder returns a new Store, so handing one to an operation cannot be
    undone by what the operation does with it.
    """
    root: Path
    bound: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def __call__(self, art: Artifact, **params) -> Handle:
        return Handle(art, self.root, {**self.bound, **params})

    def with_params(self, **params) -> "Store":
        """Bind parameter values. Selecting a locus is `with_params(locus=...)`."""
        return Store(self.root, {**self.bound, **params})

    def with_root(self, *parts) -> "Store":
        """Bind a deeper root: `with_root("operations", op.name, locus)`.

        Every part is a literal, non-empty directory name. A part is never a glob: reading
        one operation output across all loci is an iteration over locus-bound datasets, not
        a `*` segment. Falsy parts used to be skipped, for the locus-agnostic operations that
        no longer exist — and a skip is indistinguishable from a bug, since a `None` locus
        would have quietly rooted the pass one level up, on top of the op's own directory.
        """
        kept = [str(part or "") for part in parts]   # None collapses to "", caught below
        bad = [p for p in kept if not p or "*" in p or "?" in p]
        if bad:
            raise ValueError(f"root segments are non-empty literal directory names: {bad}")
        return Store(self.root.joinpath(*kept), dict(self.bound))

    def clear(self) -> None:
        """rmtree the root. Does not recreate; the next write mkdir -p's what it needs."""
        log.info("clearing store root %s", self.root)
        shutil.rmtree(self.root, ignore_errors=True)

    def create_tree(self, artifacts: Iterable[Artifact]) -> None:
        """Make the directories `artifacts` require. Derived from templates, never declared,
        so the tree cannot drift from the layout."""
        for d in required_dirs(self.root, artifacts):
            d.mkdir(parents=True, exist_ok=True)


def required_dirs(root: Union[str, Path], artifacts: Iterable[Artifact]) -> list[Path]:
    """The directories a tree must have to hold `artifacts`: the home of every artifact
    whose absence is an error.

    `on_missing` already says which those are, so this needs no second list to fall out of
    step with. An optional artifact gets its directory on first write; demanding one here
    would call a bulk dataset malformed for lacking a single-cell table.

    Takes its artifact list rather than importing one, so this layer stays below the layout
    layer. It is both what builds a new tree and what validates an existing one — the two
    cannot disagree about which directories matter.
    """
    root = Path(root)
    return sorted({Handle(a, root).dir() for a in artifacts
                   if a.on_missing is OnMissing.RAISE})
