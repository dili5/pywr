from pywr.flood.routing import LagRouter, MuskingumRouter, LinearReservoirRouter


def test_lag_router():
    r = LagRouter(lag_seconds=3600)
    dt = 3600.0
    # 1-step lag with dt=3600
    assert r.step(10.0, dt, commit=True) == 0.0
    assert r.step(20.0, dt, commit=True) == 10.0
    assert r.step(0.0, dt, commit=True) == 20.0


def test_muskingum_constant_flow_stays_constant():
    r = MuskingumRouter(K_seconds=7200.0, X=0.2)
    dt = 3600.0
    q = 50.0
    # After initialisation, constant inflow should remain constant.
    out0 = r.step(q, dt, commit=True)
    out1 = r.step(q, dt, commit=True)
    out2 = r.step(q, dt, commit=True)
    assert abs(out0 - q) < 1e-9
    assert abs(out1 - q) < 1e-6
    assert abs(out2 - q) < 1e-6


def test_linear_reservoir_step_response():
    r = LinearReservoirRouter(K_seconds=7200.0)
    dt = 3600.0
    q1 = r.step(100.0, dt, commit=True)
    q2 = r.step(100.0, dt, commit=True)
    assert 0.0 < q1 < q2 < 100.0

