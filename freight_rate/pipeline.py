"""End-to-end steps: explore, validate, train and predict, score, report."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import eda
from . import evaluation as ev
from .data import build_city_table, build_daily_market_index, clean_loads, read_loads
from .features import build_features, log_rate_per_mile
from .model import FreightRateModel
from .outliers import flag_label_outliers

log = logging.getLogger(__name__)

DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


@dataclass
class Workspace:
    raw_dev: pd.DataFrame
    raw_val: pd.DataFrame
    dev: pd.DataFrame
    val: pd.DataFrame
    fixes: dict
    daily_mi: pd.Series
    cities: pd.DataFrame
    X_dev: pd.DataFrame
    y_dev: pd.Series
    X_val: pd.DataFrame

    @property
    def month(self) -> pd.Series:
        return self.dev["date"].dt.month


def load_workspace() -> Workspace:
    raw_dev = read_loads(C.TRAIN_PATH, with_target=True)
    raw_val = read_loads(C.VALIDATION_PATH, with_target=False)
    dev, dev_fixes = clean_loads(raw_dev)
    val, val_fixes = clean_loads(raw_val)
    daily_mi = build_daily_market_index(dev, val)
    cities = build_city_table(dev, val)
    log.info("development rows %d, validation rows %d, cities %d", len(dev), len(val), len(cities))
    return Workspace(
        raw_dev=raw_dev, raw_val=raw_val, dev=dev, val=val,
        fixes={"development": dev_fixes, "validation": val_fixes},
        daily_mi=daily_mi, cities=cities,
        X_dev=build_features(dev, daily_mi, cities), y_dev=log_rate_per_mile(dev),
        X_val=build_features(val, daily_mi, cities),
    )


def screen(ws: Workspace) -> pd.DataFrame:
    flags = flag_label_outliers(ws.X_dev, ws.y_dev)
    log.info("label screening flagged %d of %d development rows", int(flags["is_outlier"].sum()), len(flags))
    return flags


def run_eda(ws: Workspace) -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    flags = screen(ws)
    keep = ~flags["is_outlier"]
    model = FreightRateModel().fit(ws.X_dev[keep], ws.y_dev[keep])
    regimes = eda.quote_regimes(ws.dev[keep], ws.X_val, ws.val, model)
    # Only IDs and the rate comparison: enough to audit against the source file without redistributing it.
    flagged = ws.raw_dev.loc[flags.index[flags["is_outlier"]], ["load_id", "date", "posted_rate"]].assign(
        expected_rate=np.exp(flags.loc[flags["is_outlier"], "expected_log_rpm"]) * ws.dev["distance"],
        ratio_actual_to_expected=lambda d: d["posted_rate"] / d["expected_rate"],
    ).round({"expected_rate": 2, "ratio_actual_to_expected": 3})
    eda.save_tables(
        data_quality=eda.data_quality_table(ws.raw_dev, ws.raw_val, flags),
        quote_regimes=regimes,
        flagged_label_outliers=flagged,
    )
    (C.RESULTS_DIR / "cleaning_fixes.json").write_text(json.dumps(ws.fixes, indent=2))
    level = eda.daily_rate_level(ws.dev, ws.X_dev[keep], ws.y_dev[keep])
    facts = eda.key_facts(ws.dev, ws.val, keep, level, ws.daily_mi)
    (C.RESULTS_DIR / "eda_facts.json").write_text(json.dumps(facts, indent=2))
    eda.plot_quote_regimes(regimes)
    eda.plot_market_index_vs_rates(level, ws.daily_mi)
    eda.plot_outlier_residuals(flags)
    log.info("exploration outputs written to %s", C.REPORTS_DIR)


def run_validation(ws: Workspace) -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rate = ws.dev["posted_rate"]
    backtest = ev.rolling_origin_backtest(ws.X_dev, ws.y_dev, rate, ws.month)
    summary = ev.summarise_backtest(backtest)
    log.info("backtest summary (clean rows):\n%s", summary.round(3).to_string())

    keep = ~screen(ws)["is_outlier"]
    X, y, r, q, m = ws.X_dev[keep], ws.y_dev[keep], rate[keep], ws.dev.loc[keep, "quote_signal"], ws.month[keep]
    city = ev.unseen_city_holdout(X, y, r)
    city_summary = city.groupby(["design", "group"])[["MAPE", "MAE", "RMSE"]].mean()
    log.info("unseen-city hold-out:\n%s", city_summary.round(3).to_string())
    random_cv = ev.random_split_contrast(X, y, r, q)
    random_summary = random_cv.groupby("design")[["MAPE", "MAE", "RMSE", "bias_pct"]].mean()
    log.info("random K-fold (contrast only):\n%s", random_summary.round(3).to_string())
    august = ev.quote_august_check(X, y, r, q, m)
    log.info("August quote check:\n%s", august.round(3).to_string())

    eda.save_tables(backtest=backtest, backtest_summary=summary, city_holdout=city, city_holdout_summary=city_summary,
                    random_kfold=random_cv, random_kfold_summary=random_summary, quote_august_check=august)
    eda.plot_backtest(backtest)


def run_predict(ws: Workspace) -> None:
    flags = screen(ws)
    keep = ~flags["is_outlier"]
    model = FreightRateModel().fit(ws.X_dev[keep], ws.y_dev[keep])
    model.save(C.MODEL_PATH)

    predictions = write_validation_predictions(model, ws)
    december = fill_december_inputs(model, ws)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.feature_importance().to_csv(C.RESULTS_DIR / "feature_importance.csv")
    summary = {
        "training_rows": int(keep.sum()),
        "label_outliers_removed": int((~keep).sum()),
        "validation_predictions": predictions["predicted_rate"].describe().round(2).to_dict(),
        "december_predictions": december["predicted_rate"].astype(float).describe().round(2).to_dict(),
    }
    (C.RESULTS_DIR / "prediction_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s and filled %s", C.PREDICTIONS_PATH.name, C.DECEMBER_PATH.name)


def write_validation_predictions(model: FreightRateModel, ws: Workspace) -> pd.DataFrame:
    template = pd.read_csv(C.TEMPLATE_PATH, dtype={"load_id": str})
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("unexpected template columns")
    by_id = pd.Series(model.predict_rate(ws.X_val), index=ws.val["load_id"].astype(str).values)
    missing = set(template["load_id"]) - set(by_id.index)
    if missing:
        raise ValueError(f"{len(missing)} template load_ids have no validation row")
    out = template[["load_id"]].assign(predicted_rate=template["load_id"].map(by_id).round(2))
    if not (np.isfinite(out["predicted_rate"]) & (out["predicted_rate"] > 0)).all():
        raise ValueError("non-positive or non-finite predictions")
    out.to_csv(C.PREDICTIONS_PATH, index=False)
    return out


def fill_december_inputs(model: FreightRateModel, ws: Workspace) -> pd.DataFrame:
    """Fill predicted_rate in data/december_chart_inputs.csv, leaving every other column as written."""
    text = pd.read_csv(C.DECEMBER_PATH, dtype=str, keep_default_na=False)
    if list(text.columns) != DECEMBER_COLUMNS:
        raise ValueError(f"unexpected December columns {list(text.columns)}")
    rows = text.drop(columns="predicted_rate").assign(
        distance=lambda d: d["distance"].astype(float),
        weight=lambda d: d["weight"].astype(float),
        date=lambda d: pd.to_datetime(d["date"], format="%Y-%m-%d"),
    )
    rows, _ = clean_loads(rows)
    rate = model.predict_rate(build_features(rows, ws.daily_mi, ws.cities))
    text["predicted_rate"] = [f"{v:.2f}" for v in rate]
    text.to_csv(C.DECEMBER_PATH, index=False)
    return text


def run_scorer() -> None:
    cmd = [sys.executable, "score.py", "--predictions", C.PREDICTIONS_PATH.name,
           "--december-predictions", str(C.DECEMBER_PATH.relative_to(C.ROOT))]
    log.info("running %s", " ".join(cmd))
    subprocess.run(cmd, cwd=C.ROOT, check=True)
