from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math

from pywr.flood.curves import PiecewiseLinearCurve
from pywr.flood.series import TimeSeries, build_timeseries

G = 9.80665


class OutletError(ValueError):
    pass


class StageLookup:
    """Callable stage lookup by node name."""

    def __call__(self, node_name: str) -> float:  # pragma: no cover
        raise NotImplementedError


class Outlet:
    def discharge(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
    ) -> float:
        raise NotImplementedError

    def discharge_components(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
    ) -> dict[str, float] | None:
        """Optional component discharges for routing multiple outlets."""
        return None


@dataclass(slots=True)
class WeirOutlet(Outlet):
    crest_elev: float
    width: float
    Cw: float = 1.7  # broad-crested default-ish
    submergence_method: str = "villemonte"  # none | villemonte
    submergence_transition_ratio: float = 0.67
    submergence_m: float = 1.5
    submergence_n: float = 0.385

    def discharge(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
    ) -> float:
        h1 = float(stage_up) - float(self.crest_elev)
        if h1 <= 0.0:
            return 0.0
        q_free = float(self.Cw * self.width * (h1 ** 1.5))

        method = (self.submergence_method or "none").lower()
        if method in ("none", "off", "ignore"):
            return q_free

        # Tailwater head above crest
        h2 = max(0.0, float(stage_down) - float(self.crest_elev))
        if h2 <= 0.0:
            return q_free

        # If downstream is above upstream no flow
        if float(stage_down) >= float(stage_up):
            return 0.0

        r = h2 / h1  # submergence ratio
        if r <= float(self.submergence_transition_ratio):
            return q_free

        if method in ("villemonte", "vm"):
            # Villemonte-type correction (commonly used for submerged weirs):
            # Q_sub = Q_free * (1 - r^m)^n
            m = float(self.submergence_m)
            n = float(self.submergence_n)
            k = max(0.0, (1.0 - (r**m)))
            return float(q_free * (k**n))

        raise OutletError(f"Unknown weir submergence_method: {self.submergence_method!r}")


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
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
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
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
    ) -> float:
        return float(max(0.0, self.max_Q.value_at_index(t_index)))


@dataclass(slots=True)
class CompositeOutlet(Outlet):
    a: Outlet
    b: Outlet

    def discharge(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
    ) -> float:
        return float(
            self.a.discharge(
                t_index=t_index,
                stage_up=stage_up,
                stage_down=stage_down,
                dt=dt,
                stage_lookup=stage_lookup,
            )
            + self.b.discharge(
                t_index=t_index,
                stage_up=stage_up,
                stage_down=stage_down,
                dt=dt,
                stage_lookup=stage_lookup,
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
        stage_lookup: StageLookup | None = None,
    ) -> dict[str, float]:
        return {
            name: float(
                ot.discharge(
                    t_index=t_index,
                    stage_up=stage_up,
                    stage_down=stage_down,
                    dt=dt,
                    stage_lookup=stage_lookup,
                )
            )
            for name, ot in self.outlets.items()
        }

    def discharge(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
    ) -> float:
        comps = self.discharge_components(
            t_index=t_index,
            stage_up=stage_up,
            stage_down=stage_down,
            dt=dt,
            stage_lookup=stage_lookup,
        )
        return float(sum(comps.values()))


@dataclass(slots=True)
class RatingOutlet(Outlet):
    """Stage-discharge rating outlet: Q = f(stage_up)."""

    stage_discharge: PiecewiseLinearCurve
    factor: float = 1.0

    def discharge(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
    ) -> float:
        q = float(self.stage_discharge(float(stage_up)))
        return float(max(0.0, self.factor * q))


@dataclass(slots=True)
class TailwaterOutlet(Outlet):
    """Wrap an outlet with per-component tailwater specification.

    If `tailwater_ref` is set and `stage_lookup` is provided, the downstream stage is
    taken from that node. Otherwise `tailwater_constant` is used if given, else the
    passed `stage_down` is used.
    """

    outlet: Outlet
    tailwater_ref: str | None = None
    tailwater_constant: float | None = None

    def _resolve_tailwater(self, stage_down: float, stage_lookup: StageLookup | None) -> float:
        if self.tailwater_ref and stage_lookup is not None:
            try:
                tw = float(stage_lookup(self.tailwater_ref))
                if math.isfinite(tw):
                    return tw
            except Exception:
                pass
        if self.tailwater_constant is not None:
            return float(self.tailwater_constant)
        return float(stage_down)

    def discharge(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
    ) -> float:
        tw = self._resolve_tailwater(stage_down, stage_lookup)
        return float(
            self.outlet.discharge(
                t_index=t_index,
                stage_up=stage_up,
                stage_down=tw,
                dt=dt,
                stage_lookup=stage_lookup,
            )
        )

    def discharge_components(
        self,
        *,
        t_index: int,
        stage_up: float,
        stage_down: float,
        dt: float,
        stage_lookup: StageLookup | None = None,
    ) -> dict[str, float] | None:
        tw = self._resolve_tailwater(stage_down, stage_lookup)
        return self.outlet.discharge_components(
            t_index=t_index,
            stage_up=stage_up,
            stage_down=tw,
            dt=dt,
            stage_lookup=stage_lookup,
        )


def build_outlet(
    cfg: dict,
    *,
    time_index,
    series: dict[str, TimeSeries],
    excel_provider=None,
    node_name: str | None = None,
) -> Outlet:
    otype = (cfg.get("type") or "").lower()
    if otype == "weir":
        out: Outlet = WeirOutlet(
            crest_elev=float(cfg["crest_elev"]),
            width=float(cfg["width"]),
            Cw=float(cfg.get("Cw", 1.7)),
            submergence_method=str(cfg.get("submergence_method", "villemonte")),
            submergence_transition_ratio=float(cfg.get("submergence_transition_ratio", 0.67)),
            submergence_m=float(cfg.get("submergence_m", 1.5)),
            submergence_n=float(cfg.get("submergence_n", 0.385)),
        )
        return wrap_tailwater_if_needed(out, cfg)
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
        out = OrificeOutlet(
            invert_elev=float(cfg["invert_elev"]),
            width=float(cfg["width"]),
            height=float(cfg["height"]),
            Cd=float(cfg.get("Cd", 0.62)),
            n_open=n_open,
            n_open_series=n_open_ts,
            opening_height=opening_ts,
        )
        return wrap_tailwater_if_needed(out, cfg)
    if otype in ("max_release", "maxrelease"):
        mx = cfg["max_Q"]
        if isinstance(mx, str):
            mx_ts = series[mx]
        else:
            mx_ts = build_timeseries(time_index, mx, name="max_Q")
        out = MaxReleaseOutlet(max_Q=mx_ts)
        return wrap_tailwater_if_needed(out, cfg)
    if otype in ("rating", "rating_curve", "stage_discharge"):
        curve = cfg.get("stage_discharge_curve", None) or cfg.get("curve", None)
        factor = float(cfg.get("factor", 1.0))
        if curve is None:
            raise OutletError("RatingOutlet requires 'stage_discharge_curve' or 'curve'.")
        if curve == "excel" or isinstance(curve, (list, tuple)):
            if excel_provider is None or node_name is None:
                raise OutletError("RatingOutlet curve='excel' requires stage_storage_excel configuration.")
            sheet = cfg.get("excel_sheet", None) or cfg.get("sheet", None)
            z_col = cfg.get("stage_col", None)
            q_col = cfg.get("discharge_col", None)
            if isinstance(curve, (list, tuple)):
                # List form: ["sheet", ["Z","Q"]] or ["sheet","Z","Q"]
                if len(curve) == 2 and isinstance(curve[1], (list, tuple)) and len(curve[1]) == 2:
                    sheet = curve[0]
                    z_col, q_col = curve[1]
                elif len(curve) == 3:
                    sheet, z_col, q_col = curve
                else:
                    raise OutletError(
                        "RatingOutlet curve list form must be "
                        '["sheet", ["Z","Q"]] or ["sheet","Z","Q"].'
                    )
            pairs = excel_provider.get_stage_discharge_pairs(
                node_name, sheet_name=None if sheet is None else str(sheet), stage_col=z_col, discharge_col=q_col
            )
        else:
            pairs = curve
        c = PiecewiseLinearCurve.from_pairs(pairs, clamp=True, name=f"{node_name or 'rating'}.ZQ")
        out = RatingOutlet(stage_discharge=c, factor=factor)
        return wrap_tailwater_if_needed(out, cfg)
    if otype in ("group", "sum"):
        # Build a named group from all keys except configuration keys.
        allocation = str(cfg.get("allocation", "proportional")).lower()
        order = cfg.get("order", None)
        if order is not None and not isinstance(order, list):
            raise OutletError("OutletGroup 'order' must be a list of outlet names.")
        outlets: dict[str, Outlet] = {}
        for k, v in cfg.items():
            if k in ("type", "allocation", "order", "tailwater_ref", "tailwater_constant"):
                continue
            if not isinstance(v, dict):
                raise OutletError(f"OutletGroup item {k!r} must be an outlet mapping.")
            outlets[str(k)] = build_outlet(
                v,
                time_index=time_index,
                series=series,
                excel_provider=excel_provider,
                node_name=node_name,
            )
        if not outlets:
            raise OutletError("OutletGroup must contain at least one outlet definition.")
        out: Outlet = OutletGroup(outlets=outlets, allocation=allocation, order=order)
        # Per-group tailwater can still be set.
        if "tailwater_ref" in cfg or "tailwater_constant" in cfg:
            out = TailwaterOutlet(
                out,
                tailwater_ref=cfg.get("tailwater_ref", None),
                tailwater_constant=cfg.get("tailwater_constant", None),
            )
        return out
    if otype == "composite":
        # Backwards-compatibility (deprecated): expects 'a' and 'b'.
        a = build_outlet(
            cfg["a"],
            time_index=time_index,
            series=series,
            excel_provider=excel_provider,
            node_name=node_name,
        )
        b = build_outlet(
            cfg["b"],
            time_index=time_index,
            series=series,
            excel_provider=excel_provider,
            node_name=node_name,
        )
        return CompositeOutlet(a=a, b=b)
    raise OutletError(f"Unknown outlet type: {cfg.get('type')!r}")


def wrap_tailwater_if_needed(outlet: Outlet, cfg: dict) -> Outlet:
    if "tailwater_ref" in cfg or "tailwater_constant" in cfg:
        return TailwaterOutlet(
            outlet,
            tailwater_ref=cfg.get("tailwater_ref", None),
            tailwater_constant=cfg.get("tailwater_constant", None),
        )
    return outlet

