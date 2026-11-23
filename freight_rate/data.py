"""Reading, validating and cleaning the load files, plus the lookup tables built from them."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EQUIPMENT_TYPES

log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "equipment", "weight", "date", "market_index", "quote_signal",
]
TARGET = "posted_rate"
MEDIAN_CIRCUITY = 1.18  # road miles per great-circle mile in the development data


def read_loads(path: Path, with_target: bool) -> pd.DataFrame:
    """Read a load file and check that it has the expected columns, dates and unique IDs."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found - see README.md for where the data files go")
    df = pd.read_csv(path)
    expected = FEATURE_COLUMNS + ([TARGET] if with_target else [])
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns: {missing}")
    if df["load_id"].duplicated().any():
        raise ValueError(f"{Path(path).name} has duplicate load_id values")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def clean_loads(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fix the feature-level problems found during exploration.

    Rows are never dropped here, because every validation load needs a prediction.
    Label outliers are handled separately (see ``outliers.py``) and only for training.
    Returns the cleaned copy and a count of each fix for the report.
    """
    out = df.copy()
    fixes: dict[str, int] = {}

    for col in ("pickup", "delivery", "equipment"):
        out[col] = out[col].astype("string").str.strip()

    unknown = out["equipment"].notna() & ~out["equipment"].isin(EQUIPMENT_TYPES)
    fixes["unknown_equipment_set_missing"] = int(unknown.sum())
    out.loc[unknown, "equipment"] = pd.NA

    # Negative weights mirror the positive distribution exactly (sign-entry error).
    negative = out["weight"] < 0
    fixes["negative_weight_sign_flipped"] = int(negative.sum())
    out.loc[negative, "weight"] = out.loc[negative, "weight"].abs()
    fixes["weight_missing_left_for_model"] = int(out["weight"].isna().sum())

    # Distance is always present in the supplied files; fall back to geometry if it is not.
    bad_distance = out["distance"].isna() | (out["distance"] <= 0)
    fixes["distance_rebuilt_from_coordinates"] = int(bad_distance.sum())
    if bad_distance.any():
        if "pickup_lat" not in out.columns:
            raise ValueError("distance is missing or non-positive and there are no coordinates to rebuild it")
        geo = haversine_miles(out.loc[bad_distance, "pickup_lat"], out.loc[bad_distance, "pickup_lon"],
                              out.loc[bad_distance, "delivery_lat"], out.loc[bad_distance, "delivery_lon"])
        out.loc[bad_distance, "distance"] = np.maximum(geo * MEDIAN_CIRCUITY, 70.0)

    if "market_index" in out.columns:
        fixes["market_index_missing_imputed_same_day"] = int(out["market_index"].isna().sum())
    return out, fixes


def build_daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Mean market index per calendar day across every load supplied (features only, no labels).

    The date explains about 98% of the market index variance, so the same-day mean is the
    natural fill value for a load whose own index is missing.
    """
    stacked = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    daily = stacked.dropna().groupby("date")["market_index"].mean().sort_index()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily


def daily_market_index_for(dates: pd.Series, daily: pd.Series) -> pd.Series:
    """Look up the daily market index; for an uncovered date use the mean of the same weekday
    over the last four weeks available before it (or any time, if none precede it)."""
    dates = pd.to_datetime(dates)
    values = dates.map(daily)
    for day in dates[values.isna()].unique():
        same_weekday = daily[daily.index.dayofweek == pd.Timestamp(day).dayofweek]
        earlier = same_weekday[same_weekday.index < day]
        fallback = (earlier if len(earlier) else same_weekday).tail(4).mean()
        values[dates == day] = fallback
    return values.astype(float)


def market_index_with_fallback(df: pd.DataFrame, daily: pd.Series) -> pd.Series:
    """The load's own market index where present, otherwise the same-day mean."""
    same_day = daily_market_index_for(df["date"], daily)
    if "market_index" not in df.columns:
        return same_day
    return df["market_index"].astype(float).fillna(same_day)


def build_city_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) from pickups and deliveries; each city must have one location."""
    parts = []
    for f in frames:
        for end in ("pickup", "delivery"):
            parts.append(f[[end, f"{end}_lat", f"{end}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    cities = pd.concat(parts, ignore_index=True).dropna().drop_duplicates()
    conflicts = cities["city"][cities["city"].duplicated()]
    if len(conflicts):
        raise ValueError(f"cities with more than one coordinate pair: {sorted(set(conflicts))}")
    return cities.set_index("city").sort_index()
