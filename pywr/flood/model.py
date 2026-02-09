from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from pywr.flood.nodes import FloodNode, build_node
from pywr.flood.routing import Router, build_router
from pywr.flood.series import TimeSeries, build_time_index, build_timeseries


class FloodModelError(ValueError):
    pass


@dataclass(slots=True)
class FloodEdge:
    name: str
    from_node: str
    to_node: str
    router: Router
    factor: float = 1.0

    def route(self, q: float, dt: float, *, commit: bool) -> float:
        q0 = float(self.factor) * float(q)
        return float(self.router.step(q0, dt, commit=commit))


class FloodSimulationResult:
    def __init__(self, *, time_index: pd.DatetimeIndex, node_frames: dict[str, pd.DataFrame]):
        self.time_index = time_index
        self.node_frames = node_frames

    def node(self, name: str) -> pd.DataFrame:
        return self.node_frames[name]


class FloodModel:
    """Deterministic flood routing on a river network (no 1D hydrodynamics).

    The model is explicit in time with optional fixed-point iterations to handle
    simple tailwater coupling for outlets.
    """

    def __init__(
        self,
        *,
        time_index: pd.DatetimeIndex,
        dt_seconds: float,
        nodes: dict[str, FloodNode],
        edges: list[FloodEdge],
        max_iters: int = 10,
        tol_stage: float = 1e-3,
    ):
        self.time_index = time_index
        self.dt_seconds = float(dt_seconds)
        self.nodes = nodes
        self.edges = edges
        self.max_iters = int(max_iters)
        self.tol_stage = float(tol_stage)

        self._graph = nx.DiGraph()
        for n in nodes:
            self._graph.add_node(n)
        for e in edges:
            self._graph.add_edge(e.from_node, e.to_node, edge=e)

        if not nx.is_directed_acyclic_graph(self._graph):
            raise FloodModelError("FloodModel requires an acyclic directed network.")

        self._topo = list(nx.topological_sort(self._graph))

    @classmethod
    def load(cls, path: str) -> "FloodModel":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "FloodModel":
        tcfg = cfg.get("time", {})
        dt_seconds = float(tcfg.get("dt_seconds", 3600))
        time_index = build_time_index(
            start=tcfg["start"],
            end=tcfg["end"],
            dt_seconds=int(dt_seconds),
            inclusive=tcfg.get("inclusive", "left"),
        )

        series_cfg = cfg.get("series", {}) or {}
        series: dict[str, TimeSeries] = {}
        for sname, sdata in series_cfg.items():
            series[sname] = build_timeseries(time_index, sdata, name=sname)

        # Optional: load reservoir stage-storage curves from Excel.
        stage_storage_provider = None
        if cfg.get("stage_storage_excel") is not None:
            from pywr.flood.excel import build_excel_stage_storage_provider, ExcelCurveError

            try:
                stage_storage_provider = build_excel_stage_storage_provider(
                    cfg["stage_storage_excel"]
                )
            except ExcelCurveError as e:
                raise FloodModelError(str(e)) from e

        nodes_cfg = cfg.get("nodes", {})
        if not isinstance(nodes_cfg, dict) or not nodes_cfg:
            raise FloodModelError("Config must include a non-empty 'nodes' mapping.")
        nodes = {
            name: build_node(
                name,
                ncfg,
                time_index=time_index,
                series=series,
                stage_storage_provider=stage_storage_provider,
            )
            for name, ncfg in nodes_cfg.items()
        }

        edges_cfg = cfg.get("edges", [])
        edges: list[FloodEdge] = []
        for i, ecfg in enumerate(edges_cfg):
            frm = ecfg["from"]
            to = ecfg["to"]
            if frm not in nodes or to not in nodes:
                raise FloodModelError(f"Edge {frm!r}->{to!r} references unknown nodes.")
            r_cfg = ecfg.get("routing", {"type": "lag", "lag_seconds": 0})
            router = build_router(r_cfg)
            factor = float(ecfg.get("factor", 1.0))
            edges.append(
                FloodEdge(
                    name=str(ecfg.get("name", f"e{i}:{frm}->{to}")),
                    from_node=frm,
                    to_node=to,
                    router=router,
                    factor=factor,
                )
            )

        mcfg = cfg.get("model", {}) or {}
        return cls(
            time_index=time_index,
            dt_seconds=dt_seconds,
            nodes=nodes,
            edges=edges,
            max_iters=int(mcfg.get("max_iters", 10)),
            tol_stage=float(mcfg.get("tol_stage", 1e-3)),
        )

    def _one_pass(
        self,
        *,
        t_index: int,
        stage_guess: dict[str, float],
        commit: bool,
    ) -> dict[str, dict[str, float]]:
        dt = self.dt_seconds
        inflow_acc: dict[str, float] = {n: 0.0 for n in self.nodes}
        node_state: dict[str, dict[str, float]] = {}

        out_edges_by_node: dict[str, list[FloodEdge]] = {n: [] for n in self.nodes}
        for e in self.edges:
            out_edges_by_node[e.from_node].append(e)

        for nname in self._topo:
            node = self.nodes[nname]
            qin = inflow_acc[nname]
            st = node.step(
                t_index=t_index,
                dt=dt,
                inflow=qin,
                stage_guess=stage_guess,
                commit=commit,
            )
            node_state[nname] = {
                "inflow": float(st.inflow),
                "outflow": float(st.outflow),
                "stage": float(st.stage),
                "storage": float(st.storage),
            }
            qout = float(st.outflow)
            for e in out_edges_by_node.get(nname, []):
                qd = e.route(qout, dt, commit=commit)
                inflow_acc[e.to_node] += float(qd)

        return node_state

    def run(self) -> FloodSimulationResult:
        n_steps = len(self.time_index)
        node_names = list(self.nodes.keys())

        # Pre-allocate arrays
        data = {
            n: {
                "inflow": np.zeros(n_steps, dtype=float),
                "outflow": np.zeros(n_steps, dtype=float),
                "stage": np.full(n_steps, np.nan, dtype=float),
                "storage": np.full(n_steps, np.nan, dtype=float),
            }
            for n in node_names
        }

        # Initial stage guess from node states.
        stage_guess: dict[str, float] = {
            n: float(self.nodes[n].state.stage)
            for n in node_names
        }

        for ti in range(n_steps):
            # Fixed-point iterations for tailwater coupling.
            sg = dict(stage_guess)
            last_states = None
            for _ in range(max(1, self.max_iters)):
                states = self._one_pass(t_index=ti, stage_guess=sg, commit=False)
                # Update stage guesses; compute max delta over finite stages.
                deltas = []
                for n, st in states.items():
                    s_new = float(st["stage"])
                    if np.isfinite(s_new):
                        s_old = float(sg.get(n, s_new))
                        deltas.append(abs(s_new - s_old))
                        sg[n] = s_new
                last_states = states
                if deltas and max(deltas) <= self.tol_stage:
                    break

            # Commit final pass.
            final_states = self._one_pass(t_index=ti, stage_guess=sg, commit=True)
            stage_guess = {n: float(final_states[n]["stage"]) for n in node_names}

            for n in node_names:
                st = final_states[n]
                data[n]["inflow"][ti] = st["inflow"]
                data[n]["outflow"][ti] = st["outflow"]
                data[n]["stage"][ti] = st["stage"]
                data[n]["storage"][ti] = st["storage"]

        node_frames: dict[str, pd.DataFrame] = {}
        for n in node_names:
            node_frames[n] = pd.DataFrame(
                {
                    "inflow": data[n]["inflow"],
                    "outflow": data[n]["outflow"],
                    "stage": data[n]["stage"],
                    "storage": data[n]["storage"],
                },
                index=self.time_index,
            )
        return FloodSimulationResult(time_index=self.time_index, node_frames=node_frames)

