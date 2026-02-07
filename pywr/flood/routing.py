from __future__ import annotations

from dataclasses import dataclass
from typing import Deque
from collections import deque

import math


class RoutingError(ValueError):
    pass


class Router:
    """Base class for reach routing (edge) objects."""

    def step(self, inflow: float, dt: float, *, commit: bool) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


@dataclass
class LagRouter(Router):
    """Pure lag (delay) routing implemented as a shift register.

    lag_seconds is rounded to the nearest integer number of timesteps.
    """

    lag_seconds: float
    _buffer: Deque[float] | None = None
    _nlag: int = 0

    def setup(self, dt: float) -> None:
        if dt <= 0:
            raise RoutingError("dt must be > 0.")
        self._nlag = int(round(self.lag_seconds / dt))
        if self._nlag < 0:
            raise RoutingError("lag_seconds must be >= 0.")
        self._buffer = deque([0.0] * self._nlag, maxlen=self._nlag)

    def reset(self) -> None:
        if self._buffer is not None:
            self._buffer = deque([0.0] * self._nlag, maxlen=self._nlag)

    def step(self, inflow: float, dt: float, *, commit: bool) -> float:
        if self._buffer is None:
            self.setup(dt)
        if self._nlag == 0:
            return float(inflow)
        assert self._buffer is not None
        out = self._buffer[0]
        if commit:
            self._buffer.append(float(inflow))
        return float(out)


@dataclass
class MuskingumRouter(Router):
    """Muskingum routing with fixed parameters K and X.

    References
    ----------
    Qout(t) = C0*Qin(t) + C1*Qin(t-1) + C2*Qout(t-1)
    """

    K_seconds: float
    X: float
    _qin_prev: float = 0.0
    _qout_prev: float = 0.0
    _initialized: bool = False

    def reset(self) -> None:
        self._qin_prev = 0.0
        self._qout_prev = 0.0
        self._initialized = False

    def step(self, inflow: float, dt: float, *, commit: bool) -> float:
        if dt <= 0:
            raise RoutingError("dt must be > 0.")
        if self.K_seconds <= 0:
            raise RoutingError("K_seconds must be > 0.")
        if not (0.0 <= self.X <= 0.5):
            raise RoutingError("X must be within [0, 0.5].")

        denom = 2.0 * self.K_seconds * (1.0 - self.X) + dt
        c0 = (dt - 2.0 * self.K_seconds * self.X) / denom
        c1 = (dt + 2.0 * self.K_seconds * self.X) / denom
        c2 = (2.0 * self.K_seconds * (1.0 - self.X) - dt) / denom

        qin = float(inflow)
        if not self._initialized:
            # Reasonable initial condition: outflow equals inflow.
            qout = qin
        else:
            qout = c0 * qin + c1 * self._qin_prev + c2 * self._qout_prev

        if commit:
            self._qin_prev = qin
            self._qout_prev = float(qout)
            self._initialized = True
        return float(qout)


@dataclass
class LinearReservoirRouter(Router):
    """A simple linear reservoir router (exponential smoothing).

    dS/dt = Qin - Qout, with Qout = S / K
    Discretised with an exact solution over a timestep.
    """

    K_seconds: float
    _storage: float = 0.0

    def reset(self) -> None:
        self._storage = 0.0

    def step(self, inflow: float, dt: float, *, commit: bool) -> float:
        if dt <= 0:
            raise RoutingError("dt must be > 0.")
        if self.K_seconds <= 0:
            raise RoutingError("K_seconds must be > 0.")
        qin = float(inflow)
        a = math.exp(-dt / self.K_seconds)
        # S_{t+1} = S_t * a + K * Qin * (1 - a)
        s_next = self._storage * a + self.K_seconds * qin * (1.0 - a)
        qout = s_next / self.K_seconds
        if commit:
            self._storage = s_next
        return float(qout)


def build_router(cfg: dict) -> Router:
    rtype = (cfg.get("type") or "").lower()
    if rtype in ("lag", "delay"):
        return LagRouter(lag_seconds=float(cfg["lag_seconds"]))
    if rtype in ("muskingum",):
        return MuskingumRouter(K_seconds=float(cfg["K_seconds"]), X=float(cfg["X"]))
    if rtype in ("linear_reservoir", "linearreservoir", "reservoir"):
        return LinearReservoirRouter(K_seconds=float(cfg["K_seconds"]))
    raise RoutingError(f"Unknown router type: {cfg.get('type')!r}")

