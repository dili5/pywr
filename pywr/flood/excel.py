from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class ExcelCurveError(ValueError):
    pass


@dataclass(slots=True)
class HydroExcelConfig:
    """Configuration for loading Z-V-Q curves from an Excel workbook."""

    path: str
    sheet_suffix: str = "Z-V"
    sheet_joiner: str = ""
    stage_col: str | int = "Z"
    storage_col: str | int = "V"
    discharge_col: str | int = "Q"

    def sheet_name_for_node(self, node_name: str) -> str:
        return f"{node_name}{self.sheet_joiner}{self.sheet_suffix}"


class ExcelHydroCurveProvider:
    """Load stage-storage and stage-discharge curves from an Excel workbook.

    Expected layout
    ---------------
    - One sheet per reservoir, named like: `{node_name}{sheet_joiner}{sheet_suffix}`
      Default suffix is `"Z-V"`, so `石岩生态库Z-V`.
    - Each sheet contains columns for stage (Z), storage (V) and optionally discharge (Q).
      Defaults are `Z`, `V` and `Q`.
    """

    def __init__(self, cfg: HydroExcelConfig):
        self.cfg = cfg
        self._cache_zv: dict[str, list[list[float]]] = {}
        self._cache_zq: dict[str, list[list[float]]] = {}
        self._cache_df: dict[str, pd.DataFrame] = {}

    def _read_sheet(self, sheet: str, node_name: str) -> pd.DataFrame:
        if sheet in self._cache_df:
            return self._cache_df[sheet]
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
        self._cache_df[sheet] = df
        return df

    def _get_series(self, df: pd.DataFrame, col: str | int, *, kind: str, sheet: str, node_name: str) -> pd.Series:
        if isinstance(col, int):
            try:
                s = df.iloc[:, col]
            except Exception as e:
                raise ExcelCurveError(
                    f"{kind} column index {col!r} invalid in sheet {sheet!r} for node {node_name!r}."
                ) from e
            return s
        if col not in df.columns:
            raise ExcelCurveError(
                f"{kind} column {col!r} not found in sheet {sheet!r} for node {node_name!r}."
            )
        return df[col]

    def get_xy_pairs(
        self,
        node_name: str,
        *,
        sheet_name: str,
        x_col: str | int,
        y_col: str | int,
        x_kind: str = "X",
        y_kind: str = "Y",
        min_points: int = 2,
        require_strictly_increasing_x: bool = True,
    ) -> list[list[float]]:
        """Load generic (x,y) pairs from a sheet and validate x monotonicity."""
        df = self._read_sheet(sheet_name, node_name)
        x = self._get_series(df, x_col, kind=x_kind, sheet=sheet_name, node_name=node_name)
        y = self._get_series(df, y_col, kind=y_kind, sheet=sheet_name, node_name=node_name)
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        out = (
            pd.DataFrame({"X": x, "Y": y})
            .dropna()
            .astype(float)
            .sort_values("X")
        )
        if out.empty or len(out) < min_points:
            raise ExcelCurveError(
                f"Sheet {sheet_name!r} for node {node_name!r} must contain at least "
                f"{min_points} valid ({x_kind},{y_kind}) rows."
            )
        out = out.drop_duplicates(subset=["X"], keep="first")
        if require_strictly_increasing_x and not (out["X"].diff().dropna() > 0).all():
            raise ExcelCurveError(
                f"{x_kind} values in sheet {sheet_name!r} for node {node_name!r} must be strictly increasing."
            )
        return [[float(r.X), float(r.Y)] for r in out.itertuples(index=False)]

    def get_stage_storage_pairs(
        self,
        node_name: str,
        *,
        sheet_name: str | None = None,
        stage_col: str | int | None = None,
        storage_col: str | int | None = None,
    ) -> list[list[float]]:
        sheet = sheet_name or self.cfg.sheet_name_for_node(node_name)
        if sheet in self._cache_zv:
            return self._cache_zv[sheet]

        st_col = self.cfg.stage_col if stage_col is None else stage_col
        v_col = self.cfg.storage_col if storage_col is None else storage_col

        df = self._read_sheet(sheet, node_name)
        pairs = self.get_xy_pairs(
            node_name,
            sheet_name=sheet,
            x_col=st_col,
            y_col=v_col,
            x_kind="Stage",
            y_kind="Storage",
        )
        self._cache_zv[sheet] = pairs
        return pairs

    def get_stage_discharge_pairs(
        self,
        node_name: str,
        *,
        sheet_name: str | None = None,
        stage_col: str | int | None = None,
        discharge_col: str | int | None = None,
    ) -> list[list[float]]:
        sheet = sheet_name or self.cfg.sheet_name_for_node(node_name)
        if sheet in self._cache_zq:
            return self._cache_zq[sheet]

        st_col = self.cfg.stage_col if stage_col is None else stage_col
        q_col = self.cfg.discharge_col if discharge_col is None else discharge_col

        pairs = self.get_xy_pairs(
            node_name,
            sheet_name=sheet,
            x_col=st_col,
            y_col=q_col,
            x_kind="Stage",
            y_kind="Discharge",
        )
        self._cache_zq[sheet] = pairs
        return pairs


def build_excel_hydro_provider(cfg: dict[str, Any]) -> ExcelHydroCurveProvider:
    if not isinstance(cfg, dict) or "path" not in cfg:
        raise ExcelCurveError("stage_storage_excel must be a mapping containing 'path'.")
    c = HydroExcelConfig(
        path=str(cfg["path"]),
        sheet_suffix=str(cfg.get("sheet_suffix", "Z-V")),
        sheet_joiner=str(cfg.get("sheet_joiner", "")),
        stage_col=cfg.get("stage_col", "Z"),
        storage_col=cfg.get("storage_col", "V"),
        discharge_col=cfg.get("discharge_col", "Q"),
    )
    return ExcelHydroCurveProvider(c)


# Backwards-compatible names
StageStorageExcelConfig = HydroExcelConfig
ExcelStageStorageProvider = ExcelHydroCurveProvider
build_excel_stage_storage_provider = build_excel_hydro_provider

