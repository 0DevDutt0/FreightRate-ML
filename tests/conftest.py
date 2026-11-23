"""Shared synthetic data: small load tables with a known pricing formula."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate.config import EQUIPMENT_TYPES
from freight_rate.data import haversine_miles

CITIES = {
    "Alpha": (35.0, -90.0), "Bravo": (40.0, -85.0), "Charlie": (33.0, -100.0),
    "Delta": (42.0, -75.0), "Echo": (30.0, -95.0), "Foxtrot": (38.0, -110.0),
}
EQUIPMENT_PREMIUM = {"Dry Van": 0.0, "Flatbed": 0.08, "Reefer": 0.12}
FAST_PARAMS = {"objective": "regression", "n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31,
               "min_child_samples": 10, "random_state": 0, "verbose": -1}


def synthetic_loads(n: int = 2000, seed: int = 0, with_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(CITIES)
    pickup = rng.choice(names, n)
    delivery = np.array([rng.choice([c for c in names if c != p]) for p in pickup])
    p_lat, p_lon = np.array([CITIES[c] for c in pickup]).T
    d_lat, d_lon = np.array([CITIES[c] for c in delivery]).T
    distance = np.maximum(haversine_miles(p_lat, p_lon, d_lat, d_lon) * 1.2 + rng.normal(0, 5, n), 70).round(1)
    equipment = rng.choice(EQUIPMENT_TYPES, n)
    weight = rng.uniform(5_000, 47_500, n).round()
    date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    dow = date.dayofweek.to_numpy()
    market_index = (1 + 0.08 * np.sin(2 * np.pi * dow / 7) + rng.normal(0, 0.01, n)).round(5)
    df = pd.DataFrame({
        "load_id": [f"TR-{i:06d}" for i in range(1, n + 1)], "pickup": pickup, "delivery": delivery,
        "pickup_lat": p_lat, "pickup_lon": p_lon, "delivery_lat": d_lat, "delivery_lon": d_lon,
        "distance": distance, "equipment": equipment, "weight": weight, "date": date,
        "market_index": market_index, "quote_signal": rng.normal(2.0, 0.2, n).round(5),
    })
    if with_target:
        log_rpm = (0.75 - 0.13 * np.log(distance / 500) + pd.Series(equipment).map(EQUIPMENT_PREMIUM).to_numpy()
                   + 2e-6 * weight + 0.1 * np.log(market_index) + rng.normal(0, 0.02, n))
        df["posted_rate"] = (np.exp(log_rpm) * distance).round(2)
    return df


@pytest.fixture
def loads() -> pd.DataFrame:
    return synthetic_loads()
