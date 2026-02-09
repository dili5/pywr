import numpy as np

from pywr.flood.model import FloodModel


def test_component_routing_from_reservoir_outlet_group():
    cfg = {
        "time": {
            "start": "2026-01-01 00:00:00",
            "end": "2026-01-01 02:00:00",
            "dt_seconds": 3600,
            "inclusive": "left",
        },
        "series": {},
        "nodes": {
            "R": {
                "type": "reservoir",
                "initial_storage": 1.0e9,
                "stage_storage_curve": [[0.0, 0.0], [1.0, 1.0e9]],
                "outlet": {
                    "type": "sum",
                    "to_a": {"type": "max_release", "max_Q": 1.0},
                    "to_b": {"type": "max_release", "max_Q": 2.0},
                },
            },
            "A": {"type": "junction"},
            "B": {"type": "junction"},
        },
        "edges": [
            {"from": "R", "to": "A", "from_outlet": "to_a", "routing": {"type": "lag", "lag_seconds": 0}},
            {"from": "R", "to": "B", "from_outlet": "to_b", "routing": {"type": "lag", "lag_seconds": 0}},
        ],
    }
    m = FloodModel.from_dict(cfg)
    res = m.run()
    a = res.node("A")["outflow"].values
    b = res.node("B")["outflow"].values
    r = res.node("R")
    assert np.allclose(a, [1.0, 1.0])
    assert np.allclose(b, [2.0, 2.0])
    # Total outflow equals sum of components.
    assert np.allclose(r["outflow"].values, [3.0, 3.0])
    assert np.allclose(r["outflow_to_a"].values, [1.0, 1.0])
    assert np.allclose(r["outflow_to_b"].values, [2.0, 2.0])

