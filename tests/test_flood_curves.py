import pytest

from pywr.flood.curves import PiecewiseLinearCurve, StageStorageCurve, CurveError


def test_piecewise_linear_curve_basic_and_clamp():
    c = PiecewiseLinearCurve.from_pairs([(0.0, 0.0), (10.0, 20.0)], clamp=True)
    assert c(0.0) == 0.0
    assert c(5.0) == 10.0
    assert c(10.0) == 20.0
    # clamp at ends
    assert c(-1.0) == 0.0
    assert c(11.0) == 20.0


def test_inverse_requires_strictly_increasing_y():
    c = PiecewiseLinearCurve.from_pairs([(0.0, 1.0), (1.0, 1.0)], clamp=True)
    with pytest.raises(CurveError):
        c.inverse()


def test_stage_storage_curve_round_trip():
    ssv = StageStorageCurve.from_stage_storage_pairs([(10.0, 0.0), (11.0, 100.0), (12.0, 300.0)])
    v = ssv.storage_from_stage(11.5)
    assert 100.0 < v < 300.0
    s = ssv.stage_from_storage(v)
    assert 11.0 < s < 12.0

