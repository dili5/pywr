import numpy as np

from pywr.flood.model import FloodModel


def test_simple_network_lag():
    cfg = {
        "time": {"start": "2026-01-01 00:00:00", "end": "2026-01-01 03:00:00", "dt_seconds": 3600, "inclusive": "left"},
        "series": {"Qin": [0.0, 10.0, 0.0]},
        "nodes": {
            "A": {"type": "boundary", "inflow": "Qin"},
            "B": {"type": "junction"},
        },
        "edges": [
            {"from": "A", "to": "B", "routing": {"type": "lag", "lag_seconds": 3600}}
        ],
    }
    m = FloodModel.from_dict(cfg)
    res = m.run()
    b = res.node("B")
    assert np.allclose(b["outflow"].values, [0.0, 0.0, 10.0])


def test_reservoir_prescribed_release_limited_by_water():
    cfg = {
        "time": {"start": "2026-01-01 00:00:00", "end": "2026-01-01 02:00:00", "dt_seconds": 3600, "inclusive": "left"},
        "nodes": {
            "A": {"type": "boundary", "inflow": [0.0, 0.0]},
            "R": {
                "type": "reservoir",
                "initial_storage": 3600.0,  # m3
                "stage_storage_curve": [[0.0, 0.0], [1.0, 3600.0], [2.0, 7200.0]],
                "outlet": {"type": "max_release", "max_Q": 1.0},  # m3/s
            },
            "B": {"type": "junction"},
        },
        "edges": [
            {"from": "A", "to": "R", "routing": {"type": "lag", "lag_seconds": 0}},
            {"from": "R", "to": "B", "routing": {"type": "lag", "lag_seconds": 0}},
        ],
    }
    m = FloodModel.from_dict(cfg)
    res = m.run()
    r = res.node("R")
    # First step should release exactly 1 m3/s for 3600 s => 3600 m3 drained.
    assert abs(r["outflow"].iat[0] - 1.0) < 1e-9
    assert abs(r["storage"].iat[0] - 0.0) < 1e-6
    # Second step has no water left.
    assert abs(r["outflow"].iat[1] - 0.0) < 1e-9

