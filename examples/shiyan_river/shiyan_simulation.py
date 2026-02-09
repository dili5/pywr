#!/usr/bin/env python3
"""Build and run a Shiyan River flood routing model using Pywr.

This script consumes a custom JSON configuration (see shiyan_config.json)
and creates a Pywr model with stage-storage curves, gate openings and
lag routing. Output is written to CSV for quick inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from pywr.model import Model
from pywr.nodes import AggregatedNode, DelayNode, Input, Link, Output, Storage
from pywr.parameters import (
    ArrayIndexedParameter,
    ConstantParameter,
    FlowParameter,
    FunctionParameter,
    InterpolatedFlowParameter,
    InterpolatedParameter,
    InterpolatedVolumeParameter,
    ScaledProfileParameter,
)
from pywr.recorders import (
    NumpyArrayNodeRecorder,
    NumpyArrayParameterRecorder,
    NumpyArrayStorageRecorder,
)

SECONDS_PER_DAY = 86400.0

# Costs are used to guide Pywr's allocation when multiple routes are feasible.
BOUNDARY_COST = -1000.0
OUTLET_COST_BASE = -100.0
OUTLET_COST_STEP = 1.0


class StageStorageLibrary:
    def __init__(
        self,
        excel_path: str,
        stage_col: str,
        storage_col: str,
        discharge_col: Optional[str],
    ) -> None:
        self.excel_path = excel_path
        self.stage_col = stage_col
        self.storage_col = storage_col
        self.discharge_col = discharge_col
        self._cache: Dict[Tuple[str, str, str, Optional[str]], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}

    def load_curve(
        self,
        sheet_name: str,
        stage_col: Optional[str] = None,
        storage_col: Optional[str] = None,
        discharge_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        stage_col = stage_col or self.stage_col
        storage_col = storage_col or self.storage_col
        if discharge_col is None:
            discharge_col = self.discharge_col
        key = (sheet_name, stage_col, storage_col, discharge_col)
        if key in self._cache:
            return self._cache[key]

        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Stage-storage Excel not found: {self.excel_path}")

        df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
        cols = [stage_col, storage_col]
        if discharge_col and discharge_col in df.columns:
            cols.append(discharge_col)
        df = df[cols].dropna()
        if df.empty:
            raise ValueError(f"No usable rows in sheet '{sheet_name}' with columns {cols}.")

        stage = df[stage_col].to_numpy(dtype=float)
        storage = df[storage_col].to_numpy(dtype=float)
        discharge = (
            df[discharge_col].to_numpy(dtype=float)
            if discharge_col and discharge_col in df.columns
            else None
        )

        order = np.argsort(stage)
        stage = stage[order]
        storage = storage[order]
        if discharge is not None:
            discharge = discharge[order]

        self._cache[key] = (stage, storage, discharge)
        return stage, storage, discharge


class SeriesRegistry:
    def __init__(self, model: Model, series: Dict[str, Iterable[float]], expected_len: int):
        self.model = model
        self.series = series
        self.expected_len = expected_len
        self.params: Dict[str, ArrayIndexedParameter] = {}
        self.max_values: Dict[str, float] = {}
        for name, values in self.series.items():
            values = list(values)
            if len(values) != expected_len:
                raise ValueError(
                    f"Series '{name}' length {len(values)} != timesteps {expected_len}."
                )
            self.series[name] = values
            self.max_values[name] = max(values) if values else 0.0

    def get(self, name: str) -> ArrayIndexedParameter:
        if name not in self.params:
            if name not in self.series:
                raise KeyError(f"Series '{name}' not found in config.")
            values = np.asarray(self.series[name], dtype=float)
            self.params[name] = ArrayIndexedParameter(self.model, values, name=name)
        return self.params[name]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_flow_scale(config: Dict[str, Any]) -> float:
    if "flow_scale" in config:
        return float(config["flow_scale"])
    unit = str(config.get("flow_unit", "m3/s")).lower()
    if unit in {"m3/s", "m3s", "cms"}:
        return SECONDS_PER_DAY
    if unit in {"m3/day", "m3/d", "m3perday"}:
        return 1.0
    raise ValueError(f"Unsupported flow_unit: {unit}. Use flow_scale instead.")


def _parameter_value(param: Any, scenario_index: Any) -> float:
    if hasattr(param, "get_value"):
        return float(param.get_value(scenario_index))
    return float(param)


def _attach_dependency(dep: Any, param: Any) -> None:
    if hasattr(dep, "parents"):
        dep.parents.add(param)


def _make_orifice_capacity_parameter(
    model: Model,
    name: str,
    upstream_stage_param: Any,
    tailwater_stage_param: Optional[Any],
    opening_param: Any,
    opening_is_fraction: bool,
    invert_elev: float,
    width: float,
    height: float,
    Cd: float,
    n_open: int,
    max_Q: Optional[float],
    flow_scale: float,
) -> FunctionParameter:
    cfg = {
        "upstream": upstream_stage_param,
        "tailwater": tailwater_stage_param,
        "opening": opening_param,
        "opening_is_fraction": opening_is_fraction,
        "invert_elev": invert_elev,
        "width": width,
        "height": height,
        "Cd": Cd,
        "n_open": n_open,
        "max_Q": max_Q,
        "flow_scale": flow_scale,
    }

    def _calc(parent: Dict[str, Any], ts: Any, si: Any) -> float:
        upstream = _parameter_value(parent["upstream"], si)
        tailwater_param = parent["tailwater"]
        tailwater = _parameter_value(tailwater_param, si) if tailwater_param else -math.inf
        opening = _parameter_value(parent["opening"], si)
        if parent["opening_is_fraction"]:
            opening *= parent["height"]
        opening = max(0.0, min(opening, parent["height"]))
        area = parent["width"] * opening * parent["n_open"]
        if area <= 0.0:
            return 0.0
        head = upstream - max(tailwater, parent["invert_elev"])
        if head <= 0.0:
            return 0.0
        q = parent["Cd"] * area * math.sqrt(2.0 * 9.81 * head)
        if parent["max_Q"] is not None:
            q = min(q, parent["max_Q"])
        return q * parent["flow_scale"]

    param = FunctionParameter(model, cfg, _calc, name=name)
    _attach_dependency(upstream_stage_param, param)
    _attach_dependency(tailwater_stage_param, param)
    _attach_dependency(opening_param, param)
    return param


def _make_weir_capacity_parameter(
    model: Model,
    name: str,
    upstream_stage_param: Any,
    tailwater_stage_param: Optional[Any],
    crest_elev: float,
    width: float,
    Cw: float,
    submergence_method: Optional[str],
    submergence_transition_ratio: Optional[float],
    submergence_m: Optional[float],
    submergence_n: Optional[float],
    flow_scale: float,
) -> FunctionParameter:
    cfg = {
        "upstream": upstream_stage_param,
        "tailwater": tailwater_stage_param,
        "crest_elev": crest_elev,
        "width": width,
        "Cw": Cw,
        "submergence_method": submergence_method,
        "submergence_transition_ratio": submergence_transition_ratio,
        "submergence_m": submergence_m,
        "submergence_n": submergence_n,
        "flow_scale": flow_scale,
    }

    def _calc(parent: Dict[str, Any], ts: Any, si: Any) -> float:
        upstream = _parameter_value(parent["upstream"], si)
        tailwater_param = parent["tailwater"]
        tailwater = _parameter_value(tailwater_param, si) if tailwater_param else -math.inf
        h1 = upstream - parent["crest_elev"]
        if h1 <= 0.0:
            return 0.0
        q_free = parent["Cw"] * parent["width"] * (h1 ** 1.5)
        method = parent["submergence_method"]
        if not method or tailwater <= parent["crest_elev"]:
            return q_free * parent["flow_scale"]
        h2 = tailwater - parent["crest_elev"]
        ratio = h2 / h1 if h1 > 0.0 else 1.0
        if method.lower() != "villemonte":
            raise ValueError(f"Unsupported submergence method: {method}")
        transition = parent["submergence_transition_ratio"] or 0.0
        if ratio < transition:
            return q_free * parent["flow_scale"]
        m = parent["submergence_m"] or 1.5
        n = parent["submergence_n"] or 0.385
        reduction = max(1.0 - (ratio ** m), 0.0) ** n
        return q_free * reduction * parent["flow_scale"]

    param = FunctionParameter(model, cfg, _calc, name=name)
    _attach_dependency(upstream_stage_param, param)
    _attach_dependency(tailwater_stage_param, param)
    return param


def build_model(config: Dict[str, Any], excel_override: Optional[str] = None) -> Tuple[Model, Dict[str, Any], float]:
    time_cfg = config["time"]
    dt_seconds = int(time_cfg["dt_seconds"])
    freq = f"{dt_seconds}s"
    start = pd.to_datetime(time_cfg["start"])
    end = pd.to_datetime(time_cfg["end"])
    inclusive = time_cfg.get("inclusive", "both")
    times = pd.date_range(start=start, end=end, freq=freq, inclusive=inclusive)
    if times.empty:
        raise ValueError("No timesteps derived from time configuration.")

    model = Model(start=times[0], end=times[-1], timestep=freq)
    flow_scale = _resolve_flow_scale(config)

    series_registry = SeriesRegistry(
        model, config.get("series", {}), expected_len=len(model.timestepper)
    )

    stage_excel_cfg = config["stage_storage_excel"]
    excel_path = excel_override or stage_excel_cfg["path"]
    curve_library = StageStorageLibrary(
        excel_path=excel_path,
        stage_col=stage_excel_cfg["stage_col"],
        storage_col=stage_excel_cfg["storage_col"],
        discharge_col=stage_excel_cfg.get("discharge_col"),
    )

    nodes_cfg = config["nodes"]
    edges_cfg = config.get("edges", [])
    outgoing_nodes = {edge["from"] for edge in edges_cfg}

    nodes: Dict[str, Any] = {}
    reservoir_levels: Dict[str, Any] = {}
    junction_levels: Dict[str, Any] = {}
    reservoir_outlets: Dict[Tuple[str, str], Link] = {}
    reservoir_outlet_nodes: Dict[str, List[Link]] = defaultdict(list)
    reservoir_outlet_order: Dict[str, List[str]] = {}
    reservoir_qmax_params: Dict[str, Any] = {}
    pending_gates: Dict[str, Dict[str, Any]] = {}

    def connect_nodes(upstream: Any, downstream: Any) -> None:
        if isinstance(downstream, Storage):
            upstream.connect(downstream, to_slot=0)
        else:
            upstream.connect(downstream)

    # First pass: create core nodes
    for name, cfg in nodes_cfg.items():
        ntype = cfg["type"]
        if ntype == "boundary":
            inflow_series = cfg["inflow"]
            inflow_param = ScaledProfileParameter(
                model,
                flow_scale,
                series_registry.get(inflow_series),
                name=f"{name}_inflow",
            )
            node = Input(
                model,
                name=name,
                min_flow=0.0,
                max_flow=inflow_param,
                cost=BOUNDARY_COST,
            )
        elif ntype == "junction":
            if name in outgoing_nodes:
                node = Link(model, name=name)
            else:
                node = Output(model, name=name)
            if "rating_curve" in cfg:
                flows = np.array([p[0] for p in cfg["rating_curve"]], dtype=float)
                stages = np.array([p[1] for p in cfg["rating_curve"]], dtype=float)
                flows = flows * flow_scale
                stage_param = InterpolatedFlowParameter(
                    model,
                    node,
                    flows,
                    stages,
                    interp_kwargs={"bounds_error": False, "fill_value": (stages[0], stages[-1])},
                    name=f"{name}_stage",
                )
                junction_levels[name] = stage_param
        elif ntype == "reservoir":
            outlet_cfg = cfg.get("outlet", {})
            order = []
            if outlet_cfg.get("type") == "sum":
                order = list(outlet_cfg.get("order", []))
            reservoir_outlet_order[name] = order
            outputs = max(len(order), 1)

            curve_info = cfg["stage_storage_curve"]
            sheet_name = curve_info[0]
            stage_col, storage_col = curve_info[1]
            stage, storage, discharge = curve_library.load_curve(
                sheet_name, stage_col=stage_col, storage_col=storage_col
            )
            initial_stage = float(cfg["initial_stage"])
            initial_volume = float(np.interp(initial_stage, stage, storage))
            node = Storage(
                model,
                name=name,
                min_volume=float(np.min(storage)),
                max_volume=float(np.max(storage)),
                initial_volume=initial_volume,
                outputs=outputs,
            )
            level_param = InterpolatedVolumeParameter(
                model,
                node,
                volumes=storage,
                values=stage,
                interp_kwargs={"bounds_error": False, "fill_value": (stage[0], stage[-1])},
                name=f"{name}_level",
            )
            node.level = level_param
            reservoir_levels[name] = level_param
            if discharge is not None:
                discharge = discharge * flow_scale
                qmax_param = InterpolatedParameter(
                    model,
                    level_param,
                    x=stage,
                    y=discharge,
                    interp_kwargs={
                        "bounds_error": False,
                        "fill_value": (discharge[0], discharge[-1]),
                    },
                    name=f"{name}_max_discharge",
                )
                reservoir_qmax_params[name] = qmax_param

            # Lateral inflows
            laterals = cfg.get("inflows", {}).get("laterals", {})
            for lat_name, lat_cfg in laterals.items():
                series_name = lat_cfg["series"]
                factor = float(lat_cfg.get("factor", 1.0))
                lat_param = ScaledProfileParameter(
                    model,
                    flow_scale * factor,
                    series_registry.get(series_name),
                    name=f"{name}_{lat_name}_inflow",
                )
                lat_node = Input(
                    model,
                    name=f"{name}_{lat_name}",
                    min_flow=0.0,
                    max_flow=lat_param,
                    cost=BOUNDARY_COST,
                )
                connect_nodes(lat_node, node)
                nodes[lat_node.name] = lat_node
        elif ntype == "gate":
            node = Link(model, name=name)
            pending_gates[name] = cfg
        else:
            raise ValueError(f"Unsupported node type: {ntype}")

        nodes[name] = node

    def resolve_stage_param(node_name: Optional[str]) -> Optional[Any]:
        if not node_name:
            return None
        if node_name in reservoir_levels:
            return reservoir_levels[node_name]
        if node_name in junction_levels:
            return junction_levels[node_name]
        return None

    # Second pass: create reservoir outlets
    for res_name, cfg in nodes_cfg.items():
        if cfg["type"] != "reservoir":
            continue
        outlet_cfg = cfg.get("outlet")
        if not outlet_cfg:
            continue
        if outlet_cfg.get("type") != "sum":
            raise ValueError(f"Reservoir outlet must be type 'sum' for {res_name}")

        order = reservoir_outlet_order.get(res_name, [])
        res_node = nodes[res_name]
        for idx, outlet_name in enumerate(order):
            o_cfg = outlet_cfg[outlet_name]
            o_type = o_cfg["type"]
            outlet_node = Link(model, name=f"{res_name}_{outlet_name}")
            outlet_node.cost = OUTLET_COST_BASE + OUTLET_COST_STEP * idx

            if o_type == "orifice":
                tailwater_ref = o_cfg.get("tailwater_ref") or cfg.get("tailwater_ref")
                tailwater_param = resolve_stage_param(tailwater_ref)
                opening_ref = o_cfg["opening_height"]
                opening_is_fraction = bool(o_cfg.get("opening_height_is_fraction", False))
                if isinstance(opening_ref, str):
                    opening_param = series_registry.get(opening_ref)
                    if not o_cfg.get("opening_height_is_fraction"):
                        if series_registry.max_values.get(opening_ref, 0.0) <= 1.0 and o_cfg["height"] > 1.0:
                            opening_is_fraction = True
                else:
                    opening_param = ConstantParameter(model, float(opening_ref))
                    if not o_cfg.get("opening_height_is_fraction") and float(opening_ref) <= 1.0 and o_cfg["height"] > 1.0:
                        opening_is_fraction = True
                max_q = o_cfg.get("max_Q")
                if max_q is not None:
                    max_q = float(max_q)
                max_flow_param = _make_orifice_capacity_parameter(
                    model,
                    name=f"{res_name}_{outlet_name}_capacity",
                    upstream_stage_param=reservoir_levels[res_name],
                    tailwater_stage_param=tailwater_param,
                    opening_param=opening_param,
                    opening_is_fraction=opening_is_fraction,
                    invert_elev=float(o_cfg["invert_elev"]),
                    width=float(o_cfg["width"]),
                    height=float(o_cfg["height"]),
                    Cd=float(o_cfg["Cd"]),
                    n_open=int(o_cfg.get("n_open", 1)),
                    max_Q=max_q,
                    flow_scale=flow_scale,
                )
                outlet_node.max_flow = max_flow_param
            elif o_type == "weir":
                tailwater_ref = o_cfg.get("tailwater_ref") or cfg.get("tailwater_ref")
                tailwater_param = resolve_stage_param(tailwater_ref)
                max_flow_param = _make_weir_capacity_parameter(
                    model,
                    name=f"{res_name}_{outlet_name}_capacity",
                    upstream_stage_param=reservoir_levels[res_name],
                    tailwater_stage_param=tailwater_param,
                    crest_elev=float(o_cfg["crest_elev"]),
                    width=float(o_cfg["width"]),
                    Cw=float(o_cfg["Cw"]),
                    submergence_method=o_cfg.get("submergence_method"),
                    submergence_transition_ratio=o_cfg.get("submergence_transition_ratio"),
                    submergence_m=o_cfg.get("submergence_m"),
                    submergence_n=o_cfg.get("submergence_n"),
                    flow_scale=flow_scale,
                )
                outlet_node.max_flow = max_flow_param
            elif o_type == "max_release":
                max_q = float(o_cfg["max_Q"]) * flow_scale
                outlet_node.max_flow = ConstantParameter(
                    model, max_q, name=f"{res_name}_{outlet_name}_max_release"
                )
            else:
                raise ValueError(f"Unsupported outlet type '{o_type}' for {res_name}")

            res_node.connect(outlet_node, from_slot=idx)
            reservoir_outlets[(res_name, outlet_name)] = outlet_node
            reservoir_outlet_nodes[res_name].append(outlet_node)
            nodes[outlet_node.name] = outlet_node

        if res_name in reservoir_qmax_params and reservoir_outlet_nodes[res_name]:
            agg = AggregatedNode(
                model, f"{res_name}_total_outflow", reservoir_outlet_nodes[res_name]
            )
            agg.factors = [1.0] * len(reservoir_outlet_nodes[res_name])
            agg.max_flow = reservoir_qmax_params[res_name]

    # Third pass: configure gates
    incoming_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges_cfg:
        incoming_edges[edge["to"]].append(edge)

    for gate_name, g_cfg in pending_gates.items():
        upstream_edges = incoming_edges.get(gate_name, [])
        if not upstream_edges:
            raise ValueError(f"Gate '{gate_name}' has no upstream edge.")
        upstream_name = upstream_edges[0]["from"]
        upstream_node = nodes[upstream_name]

        curve_info = g_cfg["upstream_stage_curve"]
        sheet_name = curve_info[0]
        q_col, z_col = curve_info[1]
        q_values, z_values, _ = curve_library.load_curve(
            sheet_name, stage_col=q_col, storage_col=z_col, discharge_col=""
        )
        q_values = q_values * flow_scale
        flow_param = FlowParameter(model, upstream_node, name=f"{gate_name}_upstream_flow")
        stage_param = InterpolatedParameter(
            model,
            flow_param,
            x=q_values,
            y=z_values,
            interp_kwargs={"bounds_error": False, "fill_value": (z_values[0], z_values[-1])},
            name=f"{gate_name}_upstream_stage",
        )

        tailwater_param = resolve_stage_param(g_cfg.get("tailwater_ref"))
        outlet_cfg = g_cfg["outlet"]
        opening_ref = outlet_cfg["opening_height"]
        opening_is_fraction = bool(outlet_cfg.get("opening_height_is_fraction", False))
        if isinstance(opening_ref, str):
            opening_param = series_registry.get(opening_ref)
            if not outlet_cfg.get("opening_height_is_fraction"):
                if series_registry.max_values.get(opening_ref, 0.0) <= 1.0 and outlet_cfg["height"] > 1.0:
                    opening_is_fraction = True
        else:
            opening_param = ConstantParameter(model, float(opening_ref))
            if not outlet_cfg.get("opening_height_is_fraction") and float(opening_ref) <= 1.0 and outlet_cfg["height"] > 1.0:
                opening_is_fraction = True

        gate_node = nodes[gate_name]
        gate_node.cost = OUTLET_COST_BASE
        max_flow_param = _make_orifice_capacity_parameter(
            model,
            name=f"{gate_name}_capacity",
            upstream_stage_param=stage_param,
            tailwater_stage_param=tailwater_param,
            opening_param=opening_param,
            opening_is_fraction=opening_is_fraction,
            invert_elev=float(outlet_cfg["invert_elev"]),
            width=float(outlet_cfg["width"]),
            height=float(outlet_cfg["height"]),
            Cd=float(outlet_cfg["Cd"]),
            n_open=int(outlet_cfg.get("n_open", 1)),
            max_Q=None,
            flow_scale=flow_scale,
        )
        gate_node.max_flow = max_flow_param

    # Connect edges with optional delay routing
    for edge in edges_cfg:
        from_name = edge["from"]
        to_name = edge["to"]
        from_outlet = edge.get("from_outlet")
        if from_outlet:
            upstream_node = reservoir_outlets[(from_name, from_outlet)]
        else:
            upstream_node = nodes[from_name]
        downstream_node = nodes[to_name]

        lag_seconds = 0
        routing = edge.get("routing", {})
        if routing.get("type") == "lag":
            lag_seconds = int(routing.get("lag_seconds", 0))
        lag_steps = int(round(lag_seconds / dt_seconds)) if lag_seconds > 0 else 0

        if lag_steps >= 1:
            delay = DelayNode(
                model,
                name=f"{upstream_node.name}_to_{downstream_node.name}_lag",
                timesteps=lag_steps,
            )
            connect_nodes(upstream_node, delay)
            connect_nodes(delay, downstream_node)
        else:
            connect_nodes(upstream_node, downstream_node)

    return model, nodes, flow_scale


def record_and_export(model: Model, nodes: Dict[str, Any], flow_scale: float, output_path: str) -> None:
    recorders: Dict[str, Any] = {}
    flow_columns: List[str] = []
    for name, node in nodes.items():
        if isinstance(node, Storage):
            rec = NumpyArrayStorageRecorder(model, node, name=f"{name}_volume")
            recorders[rec.name] = rec
            level_param = node.level
            if level_param is not None:
                rec_level = NumpyArrayParameterRecorder(
                    model, level_param, name=f"{name}_stage"
                )
                recorders[rec_level.name] = rec_level
        else:
            rec = NumpyArrayNodeRecorder(model, node, name=f"{name}_flow")
            recorders[rec.name] = rec
            flow_columns.append(rec.name)

    model.check()
    model.run()

    index = model.timestepper.datetime_index.to_timestamp(how="s")
    data = {}
    for name, rec in recorders.items():
        values = rec.values()
        if values.ndim == 2:
            values = values[:, 0]
        data[name] = values
    df = pd.DataFrame(data, index=index)

    if flow_scale != 1.0:
        for col in flow_columns:
            df[col] = df[col] / flow_scale

    df.to_csv(output_path, index_label="time")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shiyan River flood routing model.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "shiyan_config.json"),
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--excel",
        default=None,
        help="Optional override for stage-storage Excel path.",
    )
    parser.add_argument(
        "--output",
        default="shiyan_results.csv",
        help="CSV output file path.",
    )
    args = parser.parse_args()

    config = _load_json(args.config)
    model, nodes, flow_scale = build_model(config, excel_override=args.excel)
    record_and_export(model, nodes, flow_scale, output_path=args.output)


if __name__ == "__main__":
    main()
