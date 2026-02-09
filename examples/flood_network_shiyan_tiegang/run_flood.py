from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pywr.flood import FloodModel


def main() -> None:
    cfg_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("network.json")
    )
    model = FloodModel.load(str(cfg_path))
    result = model.run()

    # Print a few key series (outflow & stage)
    for n in ["石岩生态库", "石岩水库", "铁岗水库", "西乡河"]:
        df = result.node(n)[["outflow", "stage", "storage"]]
        print(f"\n== {n} ==")
        print(df.head(8))


if __name__ == "__main__":
    main()

