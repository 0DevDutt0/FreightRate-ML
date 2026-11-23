import numpy as np
import pandas as pd
import pytest

from freight_rate.data import (
    build_city_table, build_daily_market_index, clean_loads, daily_market_index_for, market_index_with_fallback,
)


def test_negative_weight_is_sign_flipped_and_missing_weight_is_kept(loads):
    loads.loc[0, "weight"] = -31_000
    loads.loc[1, "weight"] = np.nan
    cleaned, fixes = clean_loads(loads)
    assert cleaned.loc[0, "weight"] == 31_000
    assert np.isnan(cleaned.loc[1, "weight"])
    assert fixes["negative_weight_sign_flipped"] == 1
    assert fixes["weight_missing_left_for_model"] == 1


def test_cleaning_does_not_touch_the_input_or_drop_rows(loads):
    loads.loc[0, "weight"] = -10_000
    before = loads.copy()
    cleaned, _ = clean_loads(loads)
    pd.testing.assert_frame_equal(loads, before)
    assert len(cleaned) == len(loads)


def test_unknown_equipment_becomes_missing(loads):
    loads.loc[3, "equipment"] = "Tanker"
    loads.loc[4, "equipment"] = "  Reefer "
    cleaned, fixes = clean_loads(loads)
    assert pd.isna(cleaned.loc[3, "equipment"])
    assert cleaned.loc[4, "equipment"] == "Reefer"
    assert fixes["unknown_equipment_set_missing"] == 1


def test_non_positive_distance_is_rebuilt_from_coordinates(loads):
    loads.loc[5, "distance"] = 0
    cleaned, fixes = clean_loads(loads)
    assert fixes["distance_rebuilt_from_coordinates"] == 1
    assert cleaned.loc[5, "distance"] >= 70


def test_city_table_rejects_conflicting_coordinates(loads):
    table = build_city_table(loads)
    assert set(table.index) == set(loads["pickup"]) | set(loads["delivery"])
    loads.loc[0, "pickup_lat"] += 1.0
    with pytest.raises(ValueError, match="more than one coordinate"):
        build_city_table(loads)


def test_missing_market_index_uses_same_day_mean():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-03-03"] * 3 + ["2025-03-04"]),
        "market_index": [0.8, 1.0, np.nan, 1.3],
    })
    daily = build_daily_market_index(frame)
    filled = market_index_with_fallback(frame, daily)
    assert filled.tolist() == pytest.approx([0.8, 1.0, 0.9, 1.3])


def test_uncovered_date_uses_recent_same_weekday_mean():
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    daily = pd.Series(np.where(dates.dayofweek == 0, 2.0, 1.0), index=dates)
    monday_after = pd.Series(pd.to_datetime(["2025-02-10"]))  # a Monday beyond the table
    assert daily_market_index_for(monday_after, daily).iloc[0] == pytest.approx(2.0)
