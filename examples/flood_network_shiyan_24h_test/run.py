from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pywr.flood import FloodModel


def main() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("network.json")
    model = FloodModel.load(str(cfg_path))
    res = model.run()

    for n in ["石岩河节制闸上游", "石岩生态库", "石岩水库", "茅洲河", "西乡河", "供水"]:
        df = res.node(n)
        cols = [c for c in df.columns if c.startswith("outflow") or c in ("stage", "storage", "inflow")]
        print(f"\n== {n} ==")
        print(df[cols].head(8))


if __name__ == "__main__":
    main()

