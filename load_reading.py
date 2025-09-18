# load_reading.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union
import numbers

import pandas as pd
import yaml


LOAD_CFG = Path("configurations/load_config.yaml")
CTRL_CFG = Path("configurations/controller_add_config.yaml")


@lru_cache(maxsize=1)
def _load_profile() -> Tuple[str, Union[float, pd.DataFrame, tuple]]:
    """
    Returns:
      ("constant", float_watts)
      or
      ("timeseries", (t0: pd.Timestamp, loads_per_min: pd.Series indexed by minute starting at t0))

    Falls back to controller_add_config.yaml:add_load if load_config.yaml is missing.
    """
    # 1) Try dedicated load_config.yaml
    if LOAD_CFG.exists():
        try:
            with LOAD_CFG.open("r") as f:
                y = yaml.safe_load(f) or {}
            lp = (y.get("load_profile") or {})
            mode = str(lp.get("mode", "constant")).lower()

            if mode == "timeseries":
                points = lp.get("points") or []
                df = pd.DataFrame(points)

                # ---- normalize columns
                if "load_w" not in df.columns:
                    for alt in ("load_W", "load", "value", "load_kw", "Consumption without charging [kW]"):
                        if alt in df.columns:
                            df["load_w"] = pd.to_numeric(df[alt], errors="coerce")
                            if alt == "Consumption without charging [kW]":
                                df["load_w"] = df["load_w"]
                            break

                if "time" not in df.columns:
                    for alt in ("timestamp", "Time [ISO8601]", "Time"):
                        if alt in df.columns:
                            df["time"] = df[alt]
                            break

                df["time"] = pd.to_datetime(df["time"], errors="coerce")
                df = df.dropna(subset=["time"]).copy()
                df["load_w"] = pd.to_numeric(df["load_w"], errors="coerce").fillna(0.0)
                df = df.sort_values("time")

                # ---- build contiguous 1-minute series (forward fill)
                t0 = df["time"].iloc[0].floor("min")
                t1 = df["time"].iloc[-1].floor("min")
                full_idx = pd.date_range(t0, t1, freq="min")
                s = (
                    df.set_index("time")["load_w"]
                      .resample("min")
                      .ffill()
                      .reindex(full_idx)
                      .ffill()
                )
                return ("timeseries", (t0, s))

            # constant mode
            const_val = float(lp.get("constant_load_w", 0.0))
            return ("constant", const_val)

        except Exception:
            pass

    # 2) Fallback to controller_add_config.yaml:add_load
    try:
        if CTRL_CFG.exists():
            with CTRL_CFG.open("r") as f:
                y = yaml.safe_load(f) or {}
            cfg0 = (y.get("InitializationSettings", {}).get("configs") or [{}])[0]
            return ("constant", float(cfg0.get("add_load", 0.0)))
    except Exception:
        pass

    # 3) Last resort
    return ("constant", 0.0)


def give_load_w(ts_or_minute) -> float:
    """
    Get load [W] for a given time.

    - If `ts_or_minute` is an int/float -> treated as *minute offset from 0* (use this with time_clock).
    - If `ts_or_minute` is a datetime/Timestamp/ISO string -> matched to nearest past minute (step-hold).

    If the requested minute is past the provided data, we clamp to the last value.
    """
    mode, obj = _load_profile()

    if mode == "constant":
        return float(obj)

    # timeseries: obj = (t0, Series-minute)
    t0, s = obj  # type: ignore[assignment]
    # 1) integer minute offset
    if isinstance(ts_or_minute, numbers.Number):
        idx = int(ts_or_minute)
        if idx < 0:
            idx = 0
        if idx >= len(s):
            idx = len(s) - 1
        return float(s.iloc[idx])

    # 2) datetime-like
    ts = pd.to_datetime(ts_or_minute)
    # convert to minute offset
    offset = int((ts.floor("T") - t0).total_seconds() // 60)
    if offset < 0:
        offset = 0
    if offset >= len(s):
        offset = len(s) - 1
    return float(s.iloc[offset])


def clear_load_cache():
    _load_profile.cache_clear()
