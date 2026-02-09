from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from pywr.flood.curves import PiecewiseLinearCurve, StageStorageCurve
from pywr.flood.outlets import Outlet, OutletGroup, build_outlet
from pywr.flood.series import TimeSeries, build_timeseries


class NodeError(ValueError):
    pass


@dataclass(slots=True)
class NodeState:
    inflow: float = 0.0
    outflow: float = 0.0
    stage: float = float("nan")
    storage: float = float("nan")
    outflow_components: dict[str, float] = field(default_factory=dict)


class FloodNode:
    """Base class for deterministic flood nodes."""

    def __init__(self, name: str):
        self.name = name
        self.state = NodeState()

    def reset(self) -> None:
        self.state = NodeState()

    def step(
        self,
        *,
        t_index: int,
        dt: float,
        inflow: float,
        stage_guess: dict[str, float],
        commit: bool,
    ) -> NodeState:
        raise NotImplementedError


class RatingStageMixin:
    def __init__(self, *, rating_curve: PiecewiseLinearCurve | None = None):
        self._rating_curve = rating_curve

    def stage_from_discharge(self, q: float) -> float:
        if self._rating_curve is None:
            return float("nan")
        return float(self._rating_curve(max(0.0, float(q))))


class BoundaryInflowNode(FloodNode, RatingStageMixin):
    def __init__(
        self,
        name: str,
        *,
        inflow: TimeSeries,
        rating_curve: PiecewiseLinearCurve | None = None,
    ):
        FloodNode.__init__(self, name)
        RatingStageMixin.__init__(self, rating_curve=rating_curve)
        self.inflow_series = inflow

    def step(
        self,
        *,
        t_index: int,
        dt: float,
        inflow: float,
        stage_guess: dict[str, float],
        commit: bool,
    ) -> NodeState:
        q = float(max(0.0, self.inflow_series.value_at_index(t_index)))
        st = self.stage_from_discharge(q)
        ns = NodeState(inflow=q, outflow=q, stage=st, storage=float("nan"))
        if commit:
            self.state = ns
        return ns


class JunctionNode(FloodNode, RatingStageMixin):
    def __init__(
        self,
        name: str,
        *,
        lateral_inflow: TimeSeries | None = None,
        rating_curve: PiecewiseLinearCurve | None = None,
    ):
        FloodNode.__init__(self, name)
        RatingStageMixin.__init__(self, rating_curve=rating_curve)
        self.lateral_inflow = lateral_inflow

    def step(
        self,
        *,
        t_index: int,
        dt: float,
        inflow: float,
        stage_guess: dict[str, float],
        commit: bool,
    ) -> NodeState:
        qlat = 0.0 if self.lateral_inflow is None else self.lateral_inflow.value_at_index(t_index)
        q = float(max(0.0, inflow + qlat))
        st = self.stage_from_discharge(q)
        ns = NodeState(inflow=q, outflow=q, stage=st, storage=float("nan"))
        if commit:
            self.state = ns
        return ns


class GateNode(FloodNode):
    """Capacity-limited pass-through node (e.g. control gate).

    This node does not store water. It computes a time-varying capacity Qmax and passes
    Qout = min(Qin, Qmax).
    """

    def __init__(
        self,
        name: str,
        *,
        outlet: Outlet,
        upstream_stage_curve: PiecewiseLinearCurve | None = None,  # Qin->Z
        upstream_stage_constant: float | None = None,
        tailwater_ref: str | None = None,
        tailwater_constant: float = 0.0,
    ):
        super().__init__(name)
        self.outlet = outlet
        self.upstream_stage_curve = upstream_stage_curve
        self.upstream_stage_constant = upstream_stage_constant
        self.tailwater_ref = tailwater_ref
        self.tailwater_constant = float(tailwater_constant)

    def _tailwater_stage(self, stage_guess: dict[str, float]) -> float:
        if self.tailwater_ref:
            return float(stage_guess.get(self.tailwater_ref, self.tailwater_constant))
        return self.tailwater_constant

    def _upstream_stage(self, qin: float) -> float:
        if self.upstream_stage_curve is not None:
            return float(self.upstream_stage_curve(float(max(0.0, qin))))
        if self.upstream_stage_constant is not None:
            return float(self.upstream_stage_constant)
        return float("nan")

    def step(
        self,
        *,
        t_index: int,
        dt: float,
        inflow: float,
        stage_guess: dict[str, float],
        commit: bool,
    ) -> NodeState:
        qin = max(0.0, float(inflow))
        zu = self._upstream_stage(qin)
        zd = self._tailwater_stage(stage_guess)

        qcap = float(
            self.outlet.discharge(
                t_index=t_index, stage_up=zu, stage_down=zd, dt=dt
            )
        )
        qout = min(qin, max(0.0, qcap))

        comps = self.outlet.discharge_components(
            t_index=t_index, stage_up=zu, stage_down=zd, dt=dt
        )
        if comps is None:
            comps_out: dict[str, float] = {}
        else:
            comps_out = {k: max(0.0, float(v)) for k, v in comps.items()}
            total_raw = sum(comps_out.values())
            if total_raw > 0.0 and qout < total_raw - 1e-12:
                if isinstance(self.outlet, OutletGroup) and self.outlet.allocation == "priority":
                    order = self.outlet.order or list(comps_out.keys())
                    remaining = qout
                    alloc: dict[str, float] = {k: 0.0 for k in comps_out.keys()}
                    for k in order:
                        if k not in comps_out:
                            continue
                        take = min(remaining, comps_out[k])
                        alloc[k] = take
                        remaining -= take
                        if remaining <= 0.0:
                            break
                    comps_out = alloc
                else:
                    scale = qout / total_raw
                    comps_out = {k: v * scale for k, v in comps_out.items()}

        ns = NodeState(
            inflow=qin,
            outflow=float(qout),
            stage=float(zu),
            storage=float("nan"),
            outflow_components=comps_out,
        )
        if commit:
            self.state = ns
        return ns


class ReservoirNode(FloodNode):
    def __init__(
        self,
        name: str,
        *,
        stage_storage: StageStorageCurve,
        outlet: Outlet,
        initial_storage: float | None = None,
        initial_stage: float | None = None,
        lateral_inflow: TimeSeries | None = None,
        lateral_inflows: list[tuple[str, TimeSeries, float]] | None = None,
        tailwater_ref: str | None = None,
        tailwater_constant: float = 0.0,
        max_iterations: int = 50,
        tol_storage: float = 1e-6,
    ):
        super().__init__(name)
        self.stage_storage = stage_storage
        self.outlet = outlet
        # Backwards-compatible single lateral inflow plus new multi-lateral inflows.
        self.lateral_inflow = lateral_inflow
        self.lateral_inflows = lateral_inflows or []
        self.tailwater_ref = tailwater_ref
        self.tailwater_constant = float(tailwater_constant)
        self.max_iterations = int(max_iterations)
        self.tol_storage = float(tol_storage)

        if initial_storage is None:
            if initial_stage is None:
                raise NodeError(
                    f"ReservoirNode {name!r} requires initial_storage or initial_stage."
                )
            initial_storage = self.stage_storage.storage_from_stage(float(initial_stage))
        self._storage = float(initial_storage)
        self.state.storage = self._storage
        self.state.stage = self.stage_storage.stage_from_storage(self._storage)

    def reset(self) -> None:
        super().reset()
        # keep initial storage as the last committed storage
        self.state.storage = self._storage
        self.state.stage = self.stage_storage.stage_from_storage(self._storage)

    def _tailwater_stage(self, stage_guess: dict[str, float]) -> float:
        if self.tailwater_ref:
            return float(stage_guess.get(self.tailwater_ref, self.tailwater_constant))
        return self.tailwater_constant

    def step(
        self,
        *,
        t_index: int,
        dt: float,
        inflow: float,
        stage_guess: dict[str, float],
        commit: bool,
    ) -> NodeState:
        qlat = 0.0
        if self.lateral_inflow is not None:
            qlat += float(self.lateral_inflow.value_at_index(t_index))
        if self.lateral_inflows:
            for _nm, ts, fac in self.lateral_inflows:
                qlat += float(fac) * float(ts.value_at_index(t_index))
        qin = max(0.0, float(inflow) + qlat)
        v0 = float(self._storage)

        # Upper bound if no release occurs.
        v_hi = v0 + qin * dt
        v_lo = 0.0

        tw = self._tailwater_stage(stage_guess)

        # Bisection on v1 with outflow evaluated at average stage.
        def outflow_for_v1(v1: float) -> float:
            v_avg = 0.5 * (v0 + v1)
            stage_up = self.stage_storage.stage_from_storage(v_avg)
            qcap = self.outlet.discharge(
                t_index=t_index, stage_up=stage_up, stage_down=tw, dt=dt
            )
            # Physical limit: cannot release more water than available this step.
            return min(max(0.0, float(qcap)), (v0 + qin * dt) / dt)

        # Solve continuity: v1 = v0 + (qin - qout(v1))*dt
        v1 = v_hi
        for _ in range(self.max_iterations):
            v_mid = 0.5 * (v_lo + v_hi)
            qout = outflow_for_v1(v_mid)
            v_rhs = v0 + (qin - qout) * dt
            err = v_mid - v_rhs
            if abs(err) <= self.tol_storage:
                v1 = v_mid
                break
            # If v_mid > v_rhs, we need smaller v_mid (or larger qout), move high down.
            if err > 0:
                v_hi = v_mid
            else:
                v_lo = v_mid
            v1 = v_mid

        qout = outflow_for_v1(v1)
        stage1 = self.stage_storage.stage_from_storage(v1)

        # Compute (optional) outlet components using the average stage used in routing.
        v_avg = 0.5 * (v0 + v1)
        stage_up_avg = self.stage_storage.stage_from_storage(v_avg)
        comps = self.outlet.discharge_components(
            t_index=t_index, stage_up=stage_up_avg, stage_down=tw, dt=dt
        )
        if comps is None:
            comps_out: dict[str, float] = {}
        else:
            # Ensure no negative components.
            comps_out = {k: max(0.0, float(v)) for k, v in comps.items()}
            total_raw = sum(comps_out.values())
            qout_f = float(qout)
            if total_raw > 0.0 and qout_f < total_raw - 1e-12:
                # The total was clamped (e.g. insufficient water). Allocate.
                if isinstance(self.outlet, OutletGroup) and self.outlet.allocation == "priority":
                    order = self.outlet.order or list(comps_out.keys())
                    remaining = qout_f
                    alloc: dict[str, float] = {k: 0.0 for k in comps_out.keys()}
                    for k in order:
                        if k not in comps_out:
                            continue
                        take = min(remaining, comps_out[k])
                        alloc[k] = take
                        remaining -= take
                        if remaining <= 0.0:
                            break
                    comps_out = alloc
                else:
                    # Default proportional scaling.
                    scale = qout_f / total_raw
                    comps_out = {k: v * scale for k, v in comps_out.items()}

        ns = NodeState(
            inflow=qin,
            outflow=float(qout),
            stage=float(stage1),
            storage=float(v1),
            outflow_components=comps_out,
        )
        if commit:
            self._storage = float(v1)
            self.state = ns
        return ns


def build_node(
    name: str,
    cfg: dict[str, Any],
    *,
    time_index,
    series: dict[str, TimeSeries],
    stage_storage_provider=None,
) -> FloodNode:
    ntype = (cfg.get("type") or "").lower()

    def _rating() -> PiecewiseLinearCurve | None:
        rc = cfg.get("rating_curve", None)
        if rc is None:
            return None
        return PiecewiseLinearCurve.from_pairs(rc, clamp=True, name=f"{name}.rating_curve")

    if ntype in ("boundary", "inflow", "boundary_inflow"):
        inflow_data = cfg.get("inflow", None)
        if isinstance(inflow_data, str):
            inflow_ts = series[inflow_data]
        else:
            inflow_ts = build_timeseries(time_index, inflow_data, name=f"{name}.inflow")
        return BoundaryInflowNode(name, inflow=inflow_ts, rating_curve=_rating())

    if ntype in ("gate", "controlgate", "sluice", "capacity_gate"):
        if stage_storage_provider is None:
            # still allow non-excel configs
            stage_storage_provider = None

        # upstream stage estimation (Q->Z)
        qz_spec = cfg.get("upstream_stage_curve", None) or cfg.get("upstream_stage", None)
        qz_curve = None
        if qz_spec is not None:
            # Accept direct pairs or excel list: ["sheet", ["Q","Z"]] / ["sheet","Q","Z"]
            if isinstance(qz_spec, dict) and "curve" in qz_spec:
                qz_spec = qz_spec["curve"]
            if isinstance(qz_spec, (list, tuple)) and len(qz_spec) > 0 and isinstance(qz_spec[0], (list, tuple)):
                # direct pairs
                qz_curve = PiecewiseLinearCurve.from_pairs(qz_spec, clamp=True, name=f"{name}.QZ")
            elif isinstance(qz_spec, (list, tuple)):
                if stage_storage_provider is None:
                    raise NodeError(f"{name} gate upstream_stage_curve requires stage_storage_excel.")
                if len(qz_spec) == 2 and isinstance(qz_spec[1], (list, tuple)) and len(qz_spec[1]) == 2:
                    sheet = str(qz_spec[0])
                    q_col, z_col = qz_spec[1]
                elif len(qz_spec) == 3:
                    sheet = str(qz_spec[0])
                    q_col, z_col = qz_spec[1], qz_spec[2]
                else:
                    raise NodeError(
                        f'{name} upstream_stage_curve list form must be ["sheet", ["Q","Z"]] or ["sheet","Q","Z"].'
                    )
                pairs = stage_storage_provider.get_xy_pairs(
                    name,
                    sheet_name=sheet,
                    x_col=q_col,
                    y_col=z_col,
                    x_kind="Discharge",
                    y_kind="Stage",
                )
                qz_curve = PiecewiseLinearCurve.from_pairs(pairs, clamp=True, name=f"{name}.QZ")
            else:
                raise NodeError(f"{name} upstream_stage_curve must be a pair list or excel list spec.")

        outlet_cfg = cfg.get("outlet")
        if outlet_cfg is None:
            raise NodeError(f"{name} gate requires outlet.")
        outlet = build_outlet(
            outlet_cfg,
            time_index=time_index,
            series=series,
            excel_provider=stage_storage_provider,
            node_name=name,
        )

        return GateNode(
            name,
            outlet=outlet,
            upstream_stage_curve=qz_curve,
            upstream_stage_constant=cfg.get("upstream_stage_constant", None),
            tailwater_ref=cfg.get("tailwater_ref", None),
            tailwater_constant=float(cfg.get("tailwater_constant", 0.0)),
        )

    if ntype in ("junction", "link", "node"):
        lat = cfg.get("lateral_inflow", None)
        lat_ts = None
        if lat is not None:
            lat_ts = series[lat] if isinstance(lat, str) else build_timeseries(
                time_index, lat, name=f"{name}.lateral_inflow"
            )
        # Optional: multiple laterals in a consistent structure.
        inflows_cfg = cfg.get("inflows", None)
        if inflows_cfg and isinstance(inflows_cfg, dict):
            laterals = inflows_cfg.get("laterals", inflows_cfg.get("lateral", None))
            if laterals is not None:
                # Sum them into a single series-equivalent by evaluating per timestep; for now
                # we keep only a single TimeSeries on JunctionNode, so we do not support this
                # directly (use ReservoirNode if you need to track storage).
                raise NodeError(
                    f"{name} junction does not currently support 'inflows.laterals'. "
                    "Use a 'reservoir' node or aggregate the lateral series upstream."
                )
        return JunctionNode(name, lateral_inflow=lat_ts, rating_curve=_rating())

    if ntype in ("reservoir", "storage"):
        # Optional consistent inflow structure: inflows.laterals
        lateral_inflows: list[tuple[str, TimeSeries, float]] = []
        inflows_cfg = cfg.get("inflows", None)
        if inflows_cfg and isinstance(inflows_cfg, dict):
            laterals = inflows_cfg.get("laterals", inflows_cfg.get("lateral", None))
            if laterals is not None:
                if isinstance(laterals, dict):
                    items = laterals.items()
                elif isinstance(laterals, list):
                    # list of {name, series, factor}
                    items = []
                    for i, it in enumerate(laterals):
                        if not isinstance(it, dict):
                            raise NodeError(f"{name} inflows.laterals[{i}] must be a mapping.")
                        nm = str(it.get("name", f"lat{i}"))
                        items.append((nm, it))
                else:
                    raise NodeError(f"{name} inflows.laterals must be dict or list.")

                if isinstance(laterals, dict):
                    for nm, spec in items:  # type: ignore[misc]
                        if isinstance(spec, (str, list, int, float)) or spec is None:
                            ts = series[spec] if isinstance(spec, str) else build_timeseries(
                                time_index, spec, name=f"{name}.laterals.{nm}"
                            )
                            lateral_inflows.append((str(nm), ts, 1.0))
                        elif isinstance(spec, dict):
                            sref = spec.get("series", spec.get("inflow", None))
                            fac = float(spec.get("factor", 1.0))
                            if sref is None:
                                raise NodeError(f"{name} lateral {nm!r} requires 'series'.")
                            ts = series[sref] if isinstance(sref, str) else build_timeseries(
                                time_index, sref, name=f"{name}.laterals.{nm}"
                            )
                            lateral_inflows.append((str(nm), ts, fac))
                        else:
                            raise NodeError(f"{name} lateral {nm!r} invalid spec type.")
                else:
                    for nm, spec in items:
                        sref = spec.get("series", spec.get("inflow", None))
                        fac = float(spec.get("factor", 1.0))
                        if sref is None:
                            raise NodeError(f"{name} lateral {nm!r} requires 'series'.")
                        ts = series[sref] if isinstance(sref, str) else build_timeseries(
                            time_index, sref, name=f"{name}.laterals.{nm}"
                        )
                        lateral_inflows.append((str(nm), ts, fac))

        ssv = cfg.get("stage_storage_curve")
        if ssv is None or ssv == "excel":
            if stage_storage_provider is None:
                raise NodeError(
                    f"{name} reservoir requires stage_storage_curve (or configure stage_storage_excel)."
                )
            ssv = stage_storage_provider.get_stage_storage_pairs(name)
        elif isinstance(ssv, (list, tuple)):
            # List form: ["sheet name", ["Z", "V"]] or ["sheet name", "Z", "V"]
            if stage_storage_provider is None:
                raise NodeError(
                    f"{name} reservoir requires stage_storage_curve (or configure stage_storage_excel)."
                )
            if len(ssv) == 2 and isinstance(ssv[1], (list, tuple)) and len(ssv[1]) == 2:
                sheet = ssv[0]
                s_col, v_col = ssv[1]
            elif len(ssv) == 3:
                sheet, s_col, v_col = ssv
            else:
                raise NodeError(
                    f"{name} stage_storage_curve list form must be "
                    f'["sheet", ["Z","V"]] or ["sheet","Z","V"].'
                )
            ssv = stage_storage_provider.get_stage_storage_pairs(
                name, sheet_name=str(sheet), stage_col=s_col, storage_col=v_col
            )
        elif isinstance(ssv, dict) and ("excel_sheet" in ssv or "sheet" in ssv):
            if stage_storage_provider is None:
                raise NodeError(
                    f"{name} reservoir requires stage_storage_curve (or configure stage_storage_excel)."
                )
            sheet = ssv.get("excel_sheet", None) or ssv.get("sheet", None)
            s_col = ssv.get("stage_col", None)
            v_col = ssv.get("storage_col", None)
            ssv = stage_storage_provider.get_stage_storage_pairs(
                name, sheet_name=sheet, stage_col=s_col, storage_col=v_col
            )
        stage_storage = StageStorageCurve.from_stage_storage_pairs(ssv, clamp=True)

        outlet_cfg = cfg.get("outlet")
        if outlet_cfg is None:
            raise NodeError(f"{name} reservoir requires outlet.")
        outlet = build_outlet(
            outlet_cfg,
            time_index=time_index,
            series=series,
            excel_provider=stage_storage_provider,
            node_name=name,
        )

        lat = cfg.get("lateral_inflow", None)
        lat_ts = None
        if lat is not None:
            lat_ts = series[lat] if isinstance(lat, str) else build_timeseries(
                time_index, lat, name=f"{name}.lateral_inflow"
            )

        return ReservoirNode(
            name,
            stage_storage=stage_storage,
            outlet=outlet,
            initial_storage=cfg.get("initial_storage", None),
            initial_stage=cfg.get("initial_stage", None),
            lateral_inflow=lat_ts,
            lateral_inflows=lateral_inflows,
            tailwater_ref=cfg.get("tailwater_ref", None),
            tailwater_constant=float(cfg.get("tailwater_constant", 0.0)),
        )

    raise NodeError(f"Unknown node type: {cfg.get('type')!r} for node {name!r}.")

