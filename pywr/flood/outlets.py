from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math

from pywr.flood.series import TimeSeries, build_timeseries

G = 9.80665


class OutletError(ValueError):
    pass


class Outlet:
    def discharge(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
    ) -> float:
        raise NotImplementedError


@dataclass(slots=True)
class WeirOutlet(Outlet):
    crest_elev: float
    width: float
    Cw: float = 1.7  # broad-crested default-ish

    def discharge(
        self, *, t_index: int, stage_up: float, stage_down: float, dt: float
    ) -> float:
        h = stage_up - self.crest_elev
        if h <= 0.0:
            return 0.0
        return float(self.Cw * self.width * (h ** 1.5))


@dataclass(slots=True)
class OrificeOutlet(Outlet):
    invert_elev: float
    width: float
    height: float
    Cd: float = 0.62
    n_open: int = 1
    opening_height: TimeSeries | None = None  # time-varying, metres

    def _opening(self, t_index: int) -> float:
        if self.opening_height is None:
            return self.height
        return float(self.opening_height.value_at_index(t_index))

    def discharge(
        self, *, t_index: int, stage_up: float, stage_down: float, dt: float
    ) -> float:
        if self.width <= 0.0 or self.height <= 0.0:
            raise OutletError("OrificeOutlet width/height must be > 0.")
        if self.n_open <= 0:
            return 0.0
        opening = max(0.0, min(self.height, self._opening(t_index)))
        if opening <= 0.0:
            return 0.0

        # Upstream head above invert.
        hu = stage_up - self.invert_elev
        if hu <= 0.0:
            return 0.0

        # Tailwater consideration (simple): if downstream below invert treat as free outfall.
        if stage_down <= self.invert_elev:
            head = hu
        else:
            head = stage_up - stage_down

        if head <= 0.0:
            return 0.0

        area = float(self.n_open) * self.width * opening
        return float(self.Cd * area * math.sqrt(2.0 * G * head))


@dataclass(slots=True)
class MaxReleaseOutlet(Outlet):
    """Release at most max_Q(t); otherwise pass inflow (no throttling)."""

    max_Q: TimeSeries

    def discharge(
        self, *, t_index: int, stage_up: float, stage_down: float, dt: float
    ) -> float:
        return float(max(0.0, self.max_Q.value_at_index(t_index)))


@dataclass(slots=True)
class CompositeOutlet(Outlet):
    a: Outlet
    b: Outlet

    def discharge(
        self, *, t_index: int, stage_up: float, stage_down: float, dt: float
    ) -> float:
        return float(
            self.a.discharge(
                t_index=t_index, stage_up=stage_up, stage_down=stage_down, dt=dt
            )
            + self.b.discharge(
                t_index=t_index, stage_up=stage_up, stage_down=stage_down, dt=dt
            )
        )


def build_outlet(
    cfg: dict,
    *,
    time_index,
    series: dict[str, TimeSeries],
) -> Outlet:
    otype = (cfg.get("type") or "").lower()
    if otype == "weir":
        return WeirOutlet(
            crest_elev=float(cfg["crest_elev"]),
            width=float(cfg["width"]),
            Cw=float(cfg.get("Cw", 1.7)),
        )
    if otype == "orifice":
        opening = cfg.get("opening_height", None)
        if isinstance(opening, str):
            opening_ts = series[opening]
        else:
            opening_ts = build_timeseries(time_index, opening, name="opening_height")
        return OrificeOutlet(
            invert_elev=float(cfg["invert_elev"]),
            width=float(cfg["width"]),
            height=float(cfg["height"]),
            Cd=float(cfg.get("Cd", 0.62)),
            n_open=int(cfg.get("n_open", 1)),
            opening_height=opening_ts,
        )
    if otype in ("max_release", "maxrelease"):
        mx = cfg["max_Q"]
        if isinstance(mx, str):
            mx_ts = series[mx]
        else:
            mx_ts = build_timeseries(time_index, mx, name="max_Q")
        return MaxReleaseOutlet(max_Q=mx_ts)
    if otype in ("composite", "sum"):
        a = build_outlet(cfg["a"], time_index=time_index, series=series)
        b = build_outlet(cfg["b"], time_index=time_index, series=series)
        return CompositeOutlet(a=a, b=b)
    raise OutletError(f"Unknown outlet type: {cfg.get('type')!r}")

