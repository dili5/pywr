"""Deterministic flood routing utilities (no 1D hydrodynamics).

This subpackage provides a lightweight river-network flood simulation engine
focused on:
- key nodes (reservoirs, junctions, boundaries)
- simplified reach routing (lag, Muskingum, linear reservoir)
- reservoir level-pool routing with simple outlets (weir/orifice/max-release)
"""

from pywr.flood.model import FloodModel, FloodSimulationResult  # noqa: F401
from pywr.flood.excel import (  # noqa: F401
    ExcelStageStorageProvider,
    StageStorageExcelConfig,
    ExcelCurveError,
)

