from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class ExcelCurveError(ValueError):
    pass


@dataclass(slots=True)
class StageStorageExcelConfig:
    """Configuration for loading stage-storage curves from an Excel workbook."""

    path: str
    sheet_suffix: str = "Z-V"
    sheet_joiner: str = ""
    stage_col: str | int = "Z"
    storage_col: str | int = "V"

    def sheet_name_for_node(self, node_name: str) -> str:
        return f"{node_name}{self.sheet_joiner}{self.sheet_suffix}"


class ExcelStageStorageProvider:
    """Load stage-storage pairs for reservoirs from an Excel workbook.

    Expected layout
    ---------------
    - One sheet per reservoir, named like: `{node_name}{sheet_joiner}{sheet_suffix}`
      Default suffix is `"Z-V"`, so `石岩生态库Z-V`.
    - Each sheet contains at least two columns for stage and storage.
      Defaults are `Z` and `V`.
    """

    def __init__(self, cfg: StageStorageExcelConfig):
        self.cfg = cfg
        self._cache: dict[str, list[list[float]]] = {}

    def get_stage_storage_pairs(
        self,
        node_name: str,
        *,
        sheet_name: str | None = None,
        stage_col: str | int | None = None,
        storage_col: str | int | None = None,
    ) -> list[list[float]]:
        sheet = sheet_name or self.cfg.sheet_name_for_node(node_name)
        if sheet in self._cache:
            return self._cache[sheet]

        st_col = self.cfg.stage_col if stage_col is None else stage_col
        v_col = self.cfg.storage_col if storage_col is None else storage_col

        try:
            df = pd.read_excel(self.cfg.path, sheet_name=sheet)
        except FileNotFoundError as e:
            raise ExcelCurveError(
                f"Excel file not found: {self.cfg.path!r} (for node {node_name!r})."
            ) from e
        except ImportError as e:
            raise ExcelCurveError(
                "Reading Excel files requires 'openpyxl'. Please install it (e.g. `pip install openpyxl`)."
            ) from e
        except ValueError as e:
            # pandas uses ValueError for missing sheets
            raise ExcelCurveError(
                f"Sheet {sheet!r} not found in {self.cfg.path!r} (for node {node_name!r})."
            ) from e
        except Exception as e:  # pragma: no cover
            raise ExcelCurveError(
                f"Failed reading {self.cfg.path!r} sheet {sheet!r}: {e}"
            ) from e

        if isinstance(st_col, int):
            z = df.iloc[:, st_col]
        else:
            if st_col not in df.columns:
                raise ExcelCurveError(
                    f"Stage column {st_col!r} not found in sheet {sheet!r} for node {node_name!r}."
                )
            z = df[st_col]
        if isinstance(v_col, int):
            v = df.iloc[:, v_col]
        else:
            if v_col not in df.columns:
                raise ExcelCurveError(
                    f"Storage column {v_col!r} not found in sheet {sheet!r} for node {node_name!r}."
                )
            v = df[v_col]

        z = pd.to_numeric(z, errors="coerce")
        v = pd.to_numeric(v, errors="coerce")
        out = (
            pd.DataFrame({"Z": z, "V": v})
            .dropna()
            .astype(float)
            .sort_values("Z")
        )
        if out.empty or len(out) < 2:
            raise ExcelCurveError(
                f"Sheet {sheet!r} for node {node_name!r} must contain at least 2 valid (Z,V) rows."
            )

        # Drop duplicate stages (keep first). Stage must be strictly increasing for interpolation.
        out = out.drop_duplicates(subset=["Z"], keep="first")
        if not (out["Z"].diff().dropna() > 0).all():
            raise ExcelCurveError(
                f"Stage values in sheet {sheet!r} for node {node_name!r} must be strictly increasing."
            )

        pairs = [[float(r.Z), float(r.V)] for r in out.itertuples(index=False)]
        self._cache[sheet] = pairs
        return pairs


def build_excel_stage_storage_provider(cfg: dict[str, Any]) -> ExcelStageStorageProvider:
    if not isinstance(cfg, dict) or "path" not in cfg:
        raise ExcelCurveError("stage_storage_excel must be a mapping containing 'path'.")
    c = StageStorageExcelConfig(
        path=str(cfg["path"]),
        sheet_suffix=str(cfg.get("sheet_suffix", "Z-V")),
        sheet_joiner=str(cfg.get("sheet_joiner", "")),
        stage_col=cfg.get("stage_col", "Z"),
        storage_col=cfg.get("storage_col", "V"),
    )
    return ExcelStageStorageProvider(c)

