import numpy as np
import pandas as pd
import pytest

from freight_rate.config import TIME_ORIGIN
from freight_rate.data import build_city_table, build_daily_market_index, clean_loads
from freight_rate.features import MARKET_TIME, build_features, log_rate_per_mile


@pytest.fixture
def tables(loads):
    cleaned, _ = clean_loads(loads)
    return cleaned, build_daily_market_index(cleaned), build_city_table(cleaned)


def test_feature_frame_has_every_model_input_and_no_quote(tables):
    cleaned, daily, cities = tables
    X = build_features(cleaned, daily, cities)
    assert set(MARKET_TIME) <= set(X.columns)
    assert "quote_signal" not in X.columns
    assert X[MARKET_TIME].drop(columns=["weight"]).notna().all().all()


def test_calendar_and_distance_features(tables):
    cleaned, daily, cities = tables
    X = build_features(cleaned, daily, cities)
    assert (X["day_of_week"] == cleaned["date"].dt.dayofweek).all()
    assert X["months_elapsed"].min() >= 0
    first = cleaned["date"] == TIME_ORIGIN
    assert (X.loc[first, "months_elapsed"] == 0).all()
    assert np.allclose(np.exp(X["log_distance"]), cleaned["distance"])


def test_december_style_rows_get_coordinates_and_market_index(tables):
    cleaned, daily, cities = tables
    day = cleaned["date"].iloc[0]
    december_like = pd.DataFrame({"pickup": [cleaned["pickup"].iloc[0]], "delivery": [cleaned["delivery"].iloc[0]],
                                  "distance": [360.0], "equipment": ["Dry Van"], "weight": [32_000.0], "date": [day]})
    X = build_features(december_like, daily, cities)
    assert X.loc[0, "pickup_lat"] == pytest.approx(cleaned["pickup_lat"].iloc[0])
    assert X.loc[0, "log_market_index"] == pytest.approx(np.log(daily[day]))


def test_unknown_city_without_coordinates_raises(tables):
    _, daily, cities = tables
    rows = pd.DataFrame({"pickup": ["Nowhere"], "delivery": ["Alpha"], "distance": [100.0], "equipment": ["Dry Van"],
                         "weight": [20_000.0], "date": [pd.Timestamp("2025-02-01")]})
    with pytest.raises(ValueError, match="Nowhere"):
        build_features(rows, daily, cities)


def test_target_is_log_rate_per_mile(loads):
    expected = np.log(loads["posted_rate"] / loads["distance"])
    assert np.allclose(log_rate_per_mile(loads), expected)
