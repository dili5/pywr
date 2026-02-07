from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class CurveError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PiecewiseLinearCurve:
    """A monotonic piecewise-linear curve with optional end clamping.

    Parameters
    ----------
    x, y:
        Sequences defining the curve points. `x` must be strictly increasing.
    clamp:
        If True (default), `__call__` clamps values outside the x-range to the
        end y-values. If False, linear extrapolation is used.
    """

    x: np.ndarray
    y: np.ndarray
    clamp: bool = True

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[Sequence[float]],
        *,
        clamp: bool = True,
        name: str | None = None,
    ) -> "PiecewiseLinearCurve":
        pairs = list(pairs)
        if len(pairs) < 2:
            raise CurveError(f"{name or 'curve'} requires at least 2 points.")
        x = np.asarray([p[0] for p in pairs], dtype=float)
        y = np.asarray([p[1] for p in pairs], dtype=float)
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise CurveError(f"{name or 'curve'} contains non-finite values.")
        if not np.all(np.diff(x) > 0):
            raise CurveError(f"{name or 'curve'} x-values must be strictly increasing.")
        return cls(x=x, y=y, clamp=clamp)

    def __call__(self, xq: float | np.ndarray) -> float | np.ndarray:
        xq_arr = np.asarray(xq, dtype=float)
        if self.clamp:
            xq_arr = np.clip(xq_arr, self.x[0], self.x[-1])
        yq = np.interp(xq_arr, self.x, self.y)
        if np.isscalar(xq):
            return float(yq)
        return yq

    def inverse(self, *, name: str | None = None) -> "PiecewiseLinearCurve":
        """Return an inverse curve assuming y is strictly increasing."""
        if not np.all(np.diff(self.y) > 0):
            raise CurveError(
                f"{name or 'curve'} inverse requires strictly increasing y-values."
            )
        return PiecewiseLinearCurve(x=self.y, y=self.x, clamp=self.clamp)


@dataclass(frozen=True, slots=True)
class StageStorageCurve:
    """Convenience wrapper for stage<->storage conversions."""

    stage_to_storage: PiecewiseLinearCurve
    storage_to_stage: PiecewiseLinearCurve

    @classmethod
    def from_stage_storage_pairs(
        cls, pairs: Iterable[Sequence[float]], *, clamp: bool = True
    ) -> "StageStorageCurve":
        s2v = PiecewiseLinearCurve.from_pairs(
            pairs, clamp=clamp, name="stage_storage"
        )
        v2s = s2v.inverse(name="stage_storage")
        return cls(stage_to_storage=s2v, storage_to_stage=v2s)

    def storage_from_stage(self, stage: float) -> float:
        return float(self.stage_to_storage(stage))

    def stage_from_storage(self, storage: float) -> float:
        return float(self.storage_to_stage(storage))

