"""`OperationRunner` — fan-out, staleness, writing, records.

Two rules:
  1. one operation x one locus = one record, and the record lives inside the directory it
     describes (`operations/<op>/<locus>/`)
  2. fan-out is runner policy — no operation overrides `loci_to_run`
"""
from __future__ import annotations

import logging
import warnings
from datetime import datetime
from time import time
from typing import List, Optional, TYPE_CHECKING

from ..structure import OPERATIONS_DIR, Store
from .base import BaseOperation
from .record import OPERATION_RECORD, OperationRecord, OutputRecord

if TYPE_CHECKING:
    from tcr_io.dataset import Dataset

log = logging.getLogger(__name__)


def _can_reuse(prior: Optional[OperationRecord], op: BaseOperation, *, force: bool) -> bool:
    """The single staleness verdict: may a prior pass's outputs stand?

    This used to be spread across `_superseded`, `_done` and two inline branches in
    `run_operation` — three predicates answering one question, which is how the
    uncleared-failed-run hole survived: a failed record was not "superseded", so nothing
    cleared its partial outputs before the retry wrote over them.

    `params` counts as identity, which closes a live bug: `FisherAssociation(grain="patient")`
    followed by `grain="repertoire"` used to warn "already complete" and hand back the
    patient-grain results.

    The version comparison is `==`, not `>=`: a DOWNGRADE is a mismatch too.

    One question, so one boolean. This returned a three-member `Rerun` enum until the third
    member turned out to BE the remaining instance of that same hole: `RUN` meant "no record,
    so compute without clearing", and a crash between `_run` and the record write leaves
    outputs with no record. `_observe` globs, so a parameterised output orphaned that way was
    adopted into the next pass's record as though that pass had produced it. Everything that
    is not a reuse now clears first — the rule failed records already followed.
    """
    return (prior is not None
            and not force
            and prior.status == "success"
            and prior.version == op.version
            and prior.params == op.params())


class OperationRunner:
    """Runs an operation over the loci it supports.

    `force` is on `__init__`, not on `run`: a runner is configured once and used for many
    operations. It is also what `rebuild` is built from.
    """

    def __init__(self, ds: "Dataset", *, force: bool = False):
        self._ds = ds
        self._force = force

    def run(self, op: BaseOperation) -> List[OperationRecord]:
        """One record per locus, so this returns a LIST.

        A locus that fails does not stop the others: it gets a failure record and the loop
        continues, which is what per-locus records make possible.
        """
        records = []
        for locus in op.loci_to_run(self._offered()):
            out = self._ds.store.with_root(OPERATIONS_DIR, op.name, locus) \
                                .with_params(locus=locus)
            if _can_reuse(OperationRecord.read(out(OPERATION_RECORD)), op, force=self._force):
                log.info("%s [%s]: reuse", op.name, locus)
                continue
            log.info("%s [%s]: run", op.name, locus)
            # Takes the record with it: the record lives inside. Unconditional — a directory
            # with no valid record can still hold files, and this is the only thing that stops
            # them being globbed into this pass's record.
            out.clear()

            ds = self._ds.select_locus(locus)
            start = time()
            try:
                op._run(ds, out)
                outputs = self._observe(op, out)     # inside the try: a declared output that
                status, error = "success", None      # was never written is a failed pass
            except Exception as e:
                warnings.warn(f"Operation {op.name} failed on locus={locus}: "
                              f"{type(e).__name__}: {e}", stacklevel=2)
                log.exception("%s failed on locus=%s", op.name, locus)
                outputs = self._observe(op, out, check=False)   # whatever the partial run left
                status, error = "failure", f"{type(e).__name__}: {e}"
            records.append(self._record(op, locus, out, outputs, status=status, error=error,
                                        duration_s=time() - start))

        if not records:
            warnings.warn(
                f"Operation {op.name} v{op.version} is already complete for every locus it "
                f"runs on; nothing to do. Pass force=True (or use ds.rerun) to recompute.",
                stacklevel=2,
            )
        return records

    def _offered(self) -> frozenset:
        """The loci this run may fan out over.

        A locus-BOUND dataset offers exactly its locus: `ds.select_locus("TRB").run_operation(op)`
        runs TRB and nothing else. That is what makes a nested run cheap — `FisherAssociation`
        needs `TabulateByVJ` for the one locus it is on, not for all seven.
        """
        return frozenset({self._ds.locus}) if self._ds.locus else self._ds.dispatch_loci

    def rebuild(self, op_cls: type, **overrides) -> List[OperationRecord]:
        """Reconstruct an op from its recorded params (+ overrides) and force it.

        Params come from any one of the op's per-locus records — they agree, because differing
        params are exactly what `_can_reuse` treats as a different run. Works because dataclass
        fields are constructor args, so `op_cls(**stored)` rebuilds the op.
        """
        prior = next((r for r in self._ds.operations if r.operation_name == op_cls.name), None)
        params = (prior.params if prior else {}) or {}
        return OperationRunner(self._ds, force=True).run(op_cls(**{**params, **overrides}))

    @staticmethod
    def _observe(op: BaseOperation, out: Store, *, check: bool = True) -> List[OutputRecord]:
        """What the pass actually wrote — each declared artifact, globbed.

        Observed rather than derived from the class, so a parameterised artifact — one table
        per filter reason, one per Fisher test — is recorded exactly as many times as it was
        written. With `check`, a declared output with no `{parameters}` is mandatory: its path
        is fixed, so a run that did not write it forgot to. Parameterised artifacts are exempt
        — "no filter reason fired" is a legitimate zero.
        """
        outputs = [
            OutputRecord(name=art.key, template=str(p.relative_to(out.root)),
                         format=art.format.name)
            for art in op.artifacts() for p in out(art).glob()
        ]
        written = {o.name for o in outputs}
        missing = [a.key for a in op.artifacts() if not a.params and a.key not in written]
        if check and missing:
            raise RuntimeError(
                f"{op.name} reported success but did not write declared output(s) {missing}"
            )
        return outputs

    def _record(self, op: BaseOperation, locus: str, out: Store,
                outputs: List[OutputRecord], *, status: str, error: Optional[str],
                duration_s: float) -> OperationRecord:
        """Write the pass's description into the directory it describes."""
        record = OperationRecord(
            operation_name=op.name, version=op.version, description=op.description,
            ran_at=datetime.now(), duration_s=duration_s,
            status=status, error=error, params=op.params(), locus=locus, outputs=outputs,
        )
        record.write(out(OPERATION_RECORD))
        return record
