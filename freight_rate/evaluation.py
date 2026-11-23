"""How the model is validated.

The validation file covers Nov-Dec 2025 and the development data Jan-Oct 2025, so the real task
is forecasting one to two months past the last label. The checks here copy that setup:

1. Rolling-origin backtest: train on months <= k, score months k+1 and k+2, for k = 3..8.
   Outlier screening is redone inside each training window, so no future label leaks into a fold.
2. Unseen-city hold-out: 8 validation cities never occur in training (12% of validation rows),
   so whole cities are held out to measure how the model handles a new city.
3. Random K-fold, run only as a contrast: it shuffles future months into training and is
   noticeably over-optimistic for this problem.
4. Quote check on August, the one development month where quote_signal behaves as it does in
   Nov-Dec (pure noise).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import SEED
from .features import MARKET_TIME, STATIC_DOW
from .model import FreightRateModel, LaneMedianBaseline
from .outliers import fit_label_checker, flag_label_outliers, flag_with_checker

log = logging.getLogger(__name__)

FINAL = "ensemble (final)"


def rate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAPE and bias are in percent of the actual rate (bias = mean signed error, negative means
    under-prediction, so |bias| <= MAPE); MAE and RMSE are in dollars."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    err = predicted - actual
    return {
        "n": int(len(actual)),
        "MAPE": float(np.mean(np.abs(err) / actual) * 100),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "bias_pct": float(np.mean(err / actual) * 100),
    }


def _fit_all_designs(X_fit: pd.DataFrame, y_fit: pd.Series, X_eval: pd.DataFrame) -> dict[str, np.ndarray]:
    preds = {"lane_median_baseline": LaneMedianBaseline().fit(X_fit, y_fit).predict_log_rpm(X_eval)}
    # The ablation shows why the raw market index needs elapsed time beside it.
    for name, feats in (("static_dow", STATIC_DOW), ("market_time", MARKET_TIME),
                        ("ablation: market index, no elapsed time", STATIC_DOW + ["log_market_index"])):
        preds[name] = FreightRateModel({name: feats}).fit(X_fit, y_fit).predict_log_rpm(X_eval)
    preds[FINAL] = (preds["static_dow"] + preds["market_time"]) / 2
    return preds


def rolling_origin_backtest(X: pd.DataFrame, y: pd.Series, rate: pd.Series, month: pd.Series,
                            origins=(3, 4, 5, 6, 7, 8), horizon: int = 2) -> pd.DataFrame:
    rows = []
    for k in origins:
        fit_mask = month <= k
        test_mask = month.between(k + 1, k + horizon)
        X_fit, y_fit = X[fit_mask], y[fit_mask]
        bad_fit = flag_label_outliers(X_fit, y_fit)["is_outlier"]
        X_fit, y_fit = X_fit[~bad_fit], y_fit[~bad_fit]
        X_test, y_test = X[test_mask], y[test_mask]
        bad_test = flag_with_checker(fit_label_checker(X_fit, y_fit), X_test, y_test)
        log.info("origin month %d: %d training rows (%d outliers removed), %d test rows",
                 k, len(X_fit), int(bad_fit.sum()), len(X_test))

        distance = np.exp(X_test["log_distance"])
        for name, p in _fit_all_designs(X_fit, y_fit, X_test).items():
            pred_rate = np.exp(p) * distance
            for h in range(1, horizon + 1):
                in_month = (month[test_mask] == k + h).to_numpy()
                for subset, sel in (("clean", in_month & ~bad_test.to_numpy()), ("all", in_month)):
                    rows.append({"origin": k, "test_month": k + h, "horizon": h, "design": name, "rows": subset,
                                 "train_rows": len(X_fit), "train_outliers_removed": int(bad_fit.sum()),
                                 **rate_metrics(rate[test_mask].to_numpy()[sel], pred_rate.to_numpy()[sel])})
    return pd.DataFrame(rows)


def summarise_backtest(results: pd.DataFrame) -> pd.DataFrame:
    clean = results[results["rows"] == "clean"]
    by_origin = clean.groupby(["design", "origin"])["MAPE"].mean().unstack()
    summary = clean.groupby("design")[["MAPE", "MAE", "RMSE", "bias_pct"]].mean()
    summary["MAPE_worst_origin"] = by_origin.max(axis=1)
    summary["MAPE_latest_origin"] = by_origin[by_origin.columns.max()]
    summary["MAPE_all_rows"] = results[results["rows"] == "all"].groupby("design")["MAPE"].mean()
    return summary.sort_values("MAPE")


def unseen_city_holdout(X: pd.DataFrame, y: pd.Series, rate: pd.Series, n_draws: int = 3,
                        n_cities: int = 8, seed: int = SEED) -> pd.DataFrame:
    """Compare coordinates-only features with coordinates plus city-ID categories on cities the
    model has never seen. Expects rows already cleaned of label outliers."""
    cities = sorted(set(X["pickup"]) | set(X["delivery"]))
    X = X.copy()
    X["pickup_id"] = pd.Categorical(X["pickup"], categories=cities)
    X["delivery_id"] = pd.Categorical(X["delivery"], categories=cities)
    with_ids = {name: feats + ["pickup_id", "delivery_id"]
                for name, feats in (("static_dow", STATIC_DOW), ("market_time", MARKET_TIME))}
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(n_draws):
        held = set(rng.choice(cities, n_cities, replace=False))
        unseen = (X["pickup"].isin(held) | X["delivery"].isin(held)).to_numpy()
        seen_ref = ~unseen & (rng.random(len(X)) < 0.15)
        fit = ~unseen & ~seen_ref
        for design, members in (("coordinates (final)", None), ("coordinates + city IDs", with_ids)):
            model = FreightRateModel(members).fit(X[fit], y[fit])
            for group, sel in (("unseen cities", unseen), ("seen cities", seen_ref)):
                pred = model.predict_rate(X[sel])
                rows.append({"draw": draw, "held_out": ", ".join(sorted(held)), "design": design, "group": group,
                             **rate_metrics(rate[sel], pred)})
        log.info("city hold-out draw %d done (%d unseen-city rows)", draw, int(unseen.sum()))
    return pd.DataFrame(rows)


def random_split_contrast(X: pd.DataFrame, y: pd.Series, rate: pd.Series, quote: pd.Series,
                          n_splits: int = 5) -> pd.DataFrame:
    """Shuffled K-fold scores, reported only to show how optimistic they are for this problem."""
    X = X.assign(log_quote=np.log(quote))
    designs = {FINAL: None, "market_time + quote_signal": {"market_time_quote": MARKET_TIME + ["log_quote"]}}
    rows = []
    for fold, (fit_idx, test_idx) in enumerate(KFold(n_splits, shuffle=True, random_state=SEED).split(X)):
        for design, members in designs.items():
            model = FreightRateModel(members).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            rows.append({"fold": fold, "design": design,
                         **rate_metrics(rate.iloc[test_idx], model.predict_rate(X.iloc[test_idx]))})
    return pd.DataFrame(rows)


def quote_august_check(X: pd.DataFrame, y: pd.Series, rate: pd.Series, quote: pd.Series,
                       month: pd.Series) -> pd.DataFrame:
    """Train on Jan-Jul, score August (quote is noise there, as in Nov-Dec), with and without quote."""
    X = X.assign(log_quote=np.log(quote))
    fit, test = month <= 7, month == 8
    rows = []
    for design, members in ((FINAL, None), ("ensemble + quote_signal", {
            "static_dow_quote": STATIC_DOW + ["log_quote"], "market_time_quote": MARKET_TIME + ["log_quote"]})):
        model = FreightRateModel(members).fit(X[fit], y[fit])
        rows.append({"design": design, **rate_metrics(rate[test], model.predict_rate(X[test]))})
    return pd.DataFrame(rows)
