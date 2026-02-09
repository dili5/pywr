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

    def discharge_components(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
    ) -> dict[str, float] | None:
        """Optional component discharges for routing multiple outlets."""
        return None


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
    n_open_series: TimeSeries | None = None  # time-varying number of openings
    opening_height: TimeSeries | None = None  # time-varying, metres

    def _n_open(self, t_index: int) -> int:
        if self.n_open_series is None:
            return int(self.n_open)
        return int(round(float(self.n_open_series.value_at_index(t_index))))

    def _opening(self, t_index: int) -> float:
        if self.opening_height is None:
            return self.height
        return float(self.opening_height.value_at_index(t_index))

    def discharge(
        self, *, t_index: int, stage_up: float, stage_down: float, dt: float
    ) -> float:
        if self.width <= 0.0 or self.height <= 0.0:
            raise OutletError("OrificeOutlet width/height must be > 0.")
        n_open = self._n_open(t_index)
        if n_open <= 0:
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

        area = float(n_open) * self.width * opening
        return float(self.Cd * area * math.sqrt(2.0 * G * head))


@dataclass(slots=True)
class MaxReleaseOutlet(Outlet):
    """A prescribed discharge time-series (subject to available water)."""

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


@dataclass(slots=True)
class OutletGroup(Outlet):
    """A group of named outlets whose discharges are summed.

    Notes
    -----
    This object exists to support routing different outlet components to different
    downstream nodes. Use `discharge_components` to obtain the component flows.
    """

    outlets: dict[str, Outlet]
    allocation: str = "proportional"  # proportional | priority
    order: list[str] | None = None

    def discharge_components(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
    ) -> dict[str, float]:
        return {
            name: float(
                ot.discharge(
                    t_index=t_index, stage_up=stage_up, stage_down=stage_down, dt=dt
                )
            )
            for name, ot in self.outlets.items()
        }

    def discharge(
        self, *, t_index: int, stage_up: float, stage_down: float, dt: float
    ) -> float:
        comps = self.discharge_components(
            t_index=t_index, stage_up=stage_up, stage_down=stage_down, dt=dt
        )
        return float(sum(comps.values()))


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
        n_open_cfg = cfg.get("n_open", 1)
        if isinstance(n_open_cfg, str):
            n_open_ts = series[n_open_cfg]
            n_open = 1
        elif isinstance(n_open_cfg, list):
            n_open_ts = build_timeseries(time_index, n_open_cfg, name="n_open")
            n_open = 1
        else:
            n_open_ts = None
            n_open = int(n_open_cfg)
        return OrificeOutlet(
            invert_elev=float(cfg["invert_elev"]),
            width=float(cfg["width"]),
            height=float(cfg["height"]),
            Cd=float(cfg.get("Cd", 0.62)),
            n_open=n_open,
            n_open_series=n_open_ts,
            opening_height=opening_ts,
        )
    if otype in ("max_release", "maxrelease"):
        mx = cfg["max_Q"]
        if isinstance(mx, str):
            mx_ts = series[mx]
        else:
            mx_ts = build_timeseries(time_index, mx, name="max_Q")
        return MaxReleaseOutlet(max_Q=mx_ts)
    if otype in ("group", "sum"):
        # Build a named group from all keys except configuration keys.
        allocation = str(cfg.get("allocation", "proportional")).lower()
        order = cfg.get("order", None)
        if order is not None and not isinstance(order, list):
            raise OutletError("OutletGroup 'order' must be a list of outlet names.")
        outlets: dict[str, Outlet] = {}
        for k, v in cfg.items():
            if k in ("type", "allocation", "order"):
                continue
            if not isinstance(v, dict):
                raise OutletError(f"OutletGroup item {k!r} must be an outlet mapping.")
            outlets[str(k)] = build_outlet(v, time_index=time_index, series=series)
        if not outlets:
            raise OutletError("OutletGroup must contain at least one outlet definition.")
        return OutletGroup(outlets=outlets, allocation=allocation, order=order)
    if otype == "composite":
        # Backwards-compatibility (deprecated): expects 'a' and 'b'.
        a = build_outlet(cfg["a"], time_index=time_index, series=series)
        b = build_outlet(cfg["b"], time_index=time_index, series=series)
        return CompositeOutlet(a=a, b=b)
    raise OutletError(f"Unknown outlet type: {cfg.get('type')!r}")

