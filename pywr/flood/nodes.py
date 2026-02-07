from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from pywr.flood.curves import PiecewiseLinearCurve, StageStorageCurve
from pywr.flood.outlets import Outlet, build_outlet
from pywr.flood.series import TimeSeries, build_timeseries


class NodeError(ValueError):
    pass


@dataclass(slots=True)
class NodeState:
    inflow: float = 0.0
    outflow: float = 0.0
    stage: float = float("nan")
    storage: float = float("nan")


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
        tailwater_ref: str | None = None,
        tailwater_constant: float = 0.0,
        max_iterations: int = 50,
        tol_storage: float = 1e-6,
    ):
        super().__init__(name)
        self.stage_storage = stage_storage
        self.outlet = outlet
        self.lateral_inflow = lateral_inflow
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
        qlat = 0.0 if self.lateral_inflow is None else float(
            self.lateral_inflow.value_at_index(t_index)
        )
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
        ns = NodeState(inflow=qin, outflow=float(qout), stage=float(stage1), storage=float(v1))
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

    if ntype in ("junction", "link", "node"):
        lat = cfg.get("lateral_inflow", None)
        lat_ts = None
        if lat is not None:
            lat_ts = series[lat] if isinstance(lat, str) else build_timeseries(
                time_index, lat, name=f"{name}.lateral_inflow"
            )
        return JunctionNode(name, lateral_inflow=lat_ts, rating_curve=_rating())

    if ntype in ("reservoir", "storage"):
        ssv = cfg.get("stage_storage_curve")
        if ssv is None:
            raise NodeError(f"{name} reservoir requires stage_storage_curve.")
        stage_storage = StageStorageCurve.from_stage_storage_pairs(ssv, clamp=True)

        outlet_cfg = cfg.get("outlet")
        if outlet_cfg is None:
            raise NodeError(f"{name} reservoir requires outlet.")
        outlet = build_outlet(outlet_cfg, time_index=time_index, series=series)

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
            tailwater_ref=cfg.get("tailwater_ref", None),
            tailwater_constant=float(cfg.get("tailwater_constant", 0.0)),
        )

    raise NodeError(f"Unknown node type: {cfg.get('type')!r} for node {name!r}.")

