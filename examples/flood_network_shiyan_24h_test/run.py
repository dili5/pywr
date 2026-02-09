from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from pywr.flood import FloodModel


def main() -> None:
    cfg_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
        else Path(__file__).with_name("network.json")
    )
    outdir = Path(__file__).with_name("outputs")
    if "--outdir" in sys.argv:
        i = sys.argv.index("--outdir")
        outdir = Path(sys.argv[i + 1])

    # Disable scientific notation; keep 3 decimals everywhere.
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    model = FloodModel.load(str(cfg_path))
    res = model.run()

    nodes = ["石岩河节制闸", "石岩生态库", "石岩水库", "茅洲河", "西乡河", "供水"]
    for n in nodes:
        df = res.node(n)
        cols = [c for c in df.columns if c.startswith("outflow") or c in ("stage", "storage", "inflow")]
        print(f"\n== {n} ==")
        print(df[cols].round(3).head(8))

    # Save CSVs with 3 decimals (no scientific notation).
    res.save_csv(str(outdir), nodes=nodes, decimals=3)
    print(f"\nSaved CSV to: {outdir}")


if __name__ == "__main__":
    main()

