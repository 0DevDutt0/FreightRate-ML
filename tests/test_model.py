import numpy as np
import pytest

from conftest import FAST_PARAMS, synthetic_loads
from freight_rate.data import build_city_table, build_daily_market_index, clean_loads
from freight_rate.evaluation import rate_metrics
from freight_rate.features import build_features, log_rate_per_mile
from freight_rate.model import FreightRateModel, LaneMedianBaseline
from freight_rate.outliers import flag_label_outliers


def prepared(n=2000, seed=0):
    cleaned, _ = clean_loads(synthetic_loads(n=n, seed=seed))
    X = build_features(cleaned, build_daily_market_index(cleaned), build_city_table(cleaned))
    return cleaned, X, log_rate_per_mile(cleaned)


def test_corrupted_labels_are_flagged_and_clean_labels_are_not():
    cleaned, X, y = prepared(n=3000)
    rng = np.random.default_rng(1)
    corrupt = rng.choice(len(y), 60, replace=False)
    factors = np.r_[np.full(30, 4.0), np.full(30, 0.25)]
    y_bad = y.copy()
    y_bad.iloc[corrupt] += np.log(factors)
    flags = flag_label_outliers(X, y_bad)["is_outlier"]
    assert flags.iloc[corrupt].all()
    assert flags.drop(flags.index[corrupt]).mean() < 0.005


def test_model_recovers_known_prices():
    _, X, y = prepared(n=3000)
    fit, test = slice(0, 2400), slice(2400, None)
    model = FreightRateModel(params=FAST_PARAMS).fit(X.iloc[fit], y.iloc[fit])
    actual = np.exp(y.iloc[test]) * np.exp(X["log_distance"].iloc[test])
    predicted = model.predict_rate(X.iloc[test])
    assert (predicted > 0).all()
    assert rate_metrics(actual, predicted)["MAPE"] < 4.0


def test_ensemble_is_the_mean_of_its_members_in_log_space():
    _, X, y = prepared()
    model = FreightRateModel(params=FAST_PARAMS).fit(X, y)
    members = model.predict_members(X)
    assert list(members.columns) == ["static_dow", "market_time"]
    assert np.allclose(model.predict_log_rpm(X), members.mean(axis=1))


def test_unfitted_model_refuses_to_predict():
    _, X, _ = prepared(n=200)
    with pytest.raises(RuntimeError):
        FreightRateModel().predict_log_rpm(X)


def test_model_round_trips_through_disk(tmp_path):
    _, X, y = prepared(n=500)
    model = FreightRateModel(params=FAST_PARAMS).fit(X, y)
    path = tmp_path / "model.joblib"
    model.save(path)
    assert np.allclose(FreightRateModel.load(path).predict_rate(X), model.predict_rate(X))


def test_lane_baseline_falls_back_for_unseen_lane():
    _, X, y = prepared(n=1000)
    baseline = LaneMedianBaseline().fit(X, y)
    new_lane = X.iloc[[0]].assign(pickup="Unknown City")
    pred = baseline.predict_log_rpm(new_lane)
    assert np.isfinite(pred).all()


def test_rate_metrics_known_values():
    m = rate_metrics(np.array([100.0, 200.0]), np.array([110.0, 190.0]))  # errors +10% and -5%
    assert m["MAPE"] == pytest.approx(7.5)
    assert m["MAE"] == pytest.approx(10.0)
    assert m["RMSE"] == pytest.approx(10.0)
    assert m["bias_pct"] == pytest.approx(2.5)
    assert abs(m["bias_pct"]) <= m["MAPE"]
