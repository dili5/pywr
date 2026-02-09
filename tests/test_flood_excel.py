import pytest
import pandas as pd

from pywr.flood.model import FloodModel


def test_stage_storage_curve_loaded_from_excel(tmp_path):
    pytest.importorskip("openpyxl")
    # Build a minimal workbook with the required sheet naming convention:
    # sheet = node_name + "Z-V", and columns are Z and V.
    xlsx = tmp_path / "Z-V-Q.xlsx"
    node_name = "石岩生态库"
    df = pd.DataFrame({"Z": [17.0, 18.0, 19.0], "V": [0.0, 800.0, 1800.0]})
    with pd.ExcelWriter(xlsx) as w:
        df.to_excel(w, sheet_name=f"{node_name}Z-V", index=False)

    cfg = {
        "time": {
            "start": "2026-01-01 00:00:00",
            "end": "2026-01-01 01:00:00",
            "dt_seconds": 3600,
            "inclusive": "left",
        },
        "stage_storage_excel": {
            "path": str(xlsx),
            "sheet_suffix": "Z-V",
            "sheet_joiner": "",
            "stage_col": "Z",
            "storage_col": "V",
        },
        "nodes": {
            node_name: {
                "type": "reservoir",
                "initial_stage": 18.0,
                # stage_storage_curve omitted -> loaded from Excel
                "outlet": {"type": "max_release", "max_Q": 0.0},
            }
        },
        "edges": [],
    }

    m = FloodModel.from_dict(cfg)
    res = m.run()
    r = res.node(node_name)
    # No inflow and no release => storage remains at initial storage (800 m3 here).
    assert abs(r["storage"].iat[0] - 800.0) < 1e-6

