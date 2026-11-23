"""Model inputs and target.

The target is log(rate per mile): the rate scales almost proportionally with distance, so
modelling the per-mile price in log space gives trees a much smoother function to learn and
turns errors into relative (percentage) errors.

Two deliberate omissions, both backed by checks in ``eda.py`` / ``evaluation.py``:
* ``quote_signal`` is not used. Its link to the rate flips by month (about +1.0 correlation in
  Jan/Feb/Mar/Jun/Sep, -0.97 in Apr/May/Jul/Oct, 0 in Aug) and it is pure noise in Nov-Dec.
* City names are not used as categories. Eight validation cities never appear in training, and
  on a city hold-out the ID features roughly add 0.6-1.0 MAPE points for unseen cities. Coordinates
  carry the same location information and still work for a new city.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAYS_PER_MONTH, EQUIPMENT_TYPES, TIME_ORIGIN
from .data import market_index_with_fallback

LANE_FEATURES = ["log_distance", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
LOAD_FEATURES = ["equipment", "weight"]
CALENDAR_FEATURES = ["day_of_week"]
MARKET_FEATURES = ["log_market_index", "months_elapsed"]

STATIC_DOW = LANE_FEATURES + LOAD_FEATURES + CALENDAR_FEATURES
MARKET_TIME = STATIC_DOW + MARKET_FEATURES


def build_features(df: pd.DataFrame, daily_market_index: pd.Series, city_table: pd.DataFrame) -> pd.DataFrame:
    """Turn cleaned load rows into model inputs.

    Rows without coordinates or a market index (the December chart file) get them from the
    city table and the daily market index, so every kind of input goes through one code path.
    The lane columns ``pickup``/``delivery`` are carried along for the lane-median baseline.
    """
    X = pd.DataFrame(index=df.index)
    for end in ("pickup", "delivery"):
        if f"{end}_lat" in df.columns:
            lat, lon = df[f"{end}_lat"], df[f"{end}_lon"]
        else:
            unknown = sorted(set(df[end]) - set(city_table.index))
            if unknown:
                raise ValueError(f"no coordinates known for {end} cities {unknown}")
            lat, lon = df[end].map(city_table["lat"]), df[end].map(city_table["lon"])
        X[f"{end}_lat"] = lat.astype(float)
        X[f"{end}_lon"] = lon.astype(float)

    X["log_distance"] = np.log(df["distance"].astype(float))
    X["equipment"] = pd.Categorical(df["equipment"].astype("object"), categories=list(EQUIPMENT_TYPES))
    X["weight"] = df["weight"].astype(float)
    X["day_of_week"] = df["date"].dt.dayofweek
    X["months_elapsed"] = (df["date"] - TIME_ORIGIN).dt.days / DAYS_PER_MONTH
    X["log_market_index"] = np.log(market_index_with_fallback(df, daily_market_index))
    X["pickup"] = df["pickup"].astype(str).values
    X["delivery"] = df["delivery"].astype(str).values
    return X


def log_rate_per_mile(df: pd.DataFrame) -> pd.Series:
    return np.log(df["posted_rate"] / df["distance"])
