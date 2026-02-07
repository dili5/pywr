from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


class SeriesError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """A simple wrapper around a pandas Series with a DatetimeIndex."""

    series: pd.Series

    def value_at_index(self, i: int) -> float:
        v = self.series.iat[i]
        try:
            return float(v)
        except Exception as e:  # pragma: no cover
            raise SeriesError(f"Invalid series value at index {i}: {v!r}") from e


def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        raise SeriesError("TimeSeries requires a pandas Series with a DatetimeIndex.")
    if s.index.has_duplicates:
        raise SeriesError("TimeSeries index contains duplicates.")
    return s.sort_index()


def build_timeseries(
    time_index: pd.DatetimeIndex,
    data: Any,
    *,
    default: float = 0.0,
    name: str | None = None,
) -> TimeSeries:
    """Build a TimeSeries from common JSON-friendly structures.

    Supported forms
    ---------------
    - None: uses `default` for all timesteps
    - number: constant for all timesteps
    - list of numbers: length must match `time_index`
    - list of [timestamp, value] pairs: will be reindexed to `time_index` with forward fill
    """
    nm = name or "series"
    if data is None:
        s = pd.Series([default] * len(time_index), index=time_index, name=nm, dtype=float)
        return TimeSeries(_ensure_datetime_index(s))
    if isinstance(data, (int, float)):
        s = pd.Series([float(data)] * len(time_index), index=time_index, name=nm, dtype=float)
        return TimeSeries(_ensure_datetime_index(s))
    if isinstance(data, list):
        if len(data) == 0:
            s = pd.Series([default] * len(time_index), index=time_index, name=nm, dtype=float)
            return TimeSeries(_ensure_datetime_index(s))
        if isinstance(data[0], (int, float)):
            if len(data) != len(time_index):
                raise SeriesError(
                    f"{nm} length {len(data)} does not match time index length {len(time_index)}."
                )
            s = pd.Series([float(v) for v in data], index=time_index, name=nm, dtype=float)
            return TimeSeries(_ensure_datetime_index(s))
        if isinstance(data[0], list) and len(data[0]) == 2:
            # pairs of [timestamp, value]
            t: list[pd.Timestamp] = []
            v: list[float] = []
            for pair in data:  # type: ignore[assignment]
                if not (isinstance(pair, list) and len(pair) == 2):
                    raise SeriesError(f"{nm} expected [timestamp, value] pairs.")
                t.append(pd.to_datetime(pair[0]))
                v.append(float(pair[1]))
            s0 = pd.Series(v, index=pd.DatetimeIndex(t), name=nm, dtype=float).sort_index()
            s = s0.reindex(time_index, method="ffill").fillna(default).astype(float)
            return TimeSeries(_ensure_datetime_index(s))
    raise SeriesError(f"Unsupported {nm} format: {type(data).__name__}")


def build_time_index(
    *,
    start: str,
    end: str,
    dt_seconds: int,
    inclusive: str = "left",
) -> pd.DatetimeIndex:
    if dt_seconds <= 0:
        raise SeriesError("dt_seconds must be > 0.")
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    freq = pd.Timedelta(seconds=int(dt_seconds))
    return pd.date_range(start=start_ts, end=end_ts, freq=freq, inclusive=inclusive)

