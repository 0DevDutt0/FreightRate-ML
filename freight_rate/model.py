"""The rate model: an average of LightGBM regressors on log(rate per mile), plus a simple baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .config import LGBM_PARAMS
from .features import MARKET_TIME, STATIC_DOW

# Two views of the market level, averaged in log space.
# * static_dow assumes the overall level reverts to its development-period average.
# * market_time adds the market index and elapsed time, so it carries the most recent regime forward.
# Backtests show each one failing in different months; the average has the lowest mean and
# worst-case error (see reports/results/backtest_summary.csv).
DEFAULT_MEMBERS: dict[str, list[str]] = {"static_dow": STATIC_DOW, "market_time": MARKET_TIME}


class FreightRateModel:
    def __init__(self, members: dict[str, list[str]] | None = None, params: dict | None = None):
        self.members = dict(members or DEFAULT_MEMBERS)
        self.params = dict(params or LGBM_PARAMS)
        self.models_: dict[str, LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FreightRateModel":
        for name, features in self.members.items():
            self.models_[name] = LGBMRegressor(**self.params).fit(X[features], y)
        return self

    def predict_members(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("model is not fitted")
        return pd.DataFrame({name: m.predict(X[self.members[name]]) for name, m in self.models_.items()},
                            index=X.index)

    def predict_log_rpm(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_members(X).mean(axis=1).to_numpy()

    def predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted total rate in dollars (rate per mile x distance)."""
        return np.exp(self.predict_log_rpm(X)) * np.exp(X["log_distance"].to_numpy())

    def feature_importance(self) -> pd.DataFrame:
        """Share of total split gain per feature, for each member."""
        frames = []
        for name, m in self.models_.items():
            gain = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=self.members[name])
            frames.append((gain / gain.sum()).rename(name))
        return pd.concat(frames, axis=1).fillna(0.0).sort_values(list(self.models_)[-1], ascending=False)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "FreightRateModel":
        return joblib.load(path)


class LaneMedianBaseline:
    """Non-ML reference: median log rate per mile for the lane and equipment seen in training,
    falling back to equipment x distance band, then to equipment alone."""

    BANDS = [0, 250, 500, 1000, 1500, 2000, 10_000]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LaneMedianBaseline":
        frame = self._keys(X).assign(y=y.to_numpy())
        self.lane_ = frame.groupby(["pickup", "delivery", "equipment"], observed=True)["y"].median()
        self.band_ = frame.groupby(["equipment", "band"], observed=True)["y"].median()
        self.equipment_ = frame.groupby("equipment", observed=True)["y"].median()
        self.overall_ = float(np.median(y))
        return self

    def predict_log_rpm(self, X: pd.DataFrame) -> np.ndarray:
        keys = self._keys(X)
        lane = pd.MultiIndex.from_frame(keys[["pickup", "delivery", "equipment"]])
        band = pd.MultiIndex.from_frame(keys[["equipment", "band"]])
        pred = pd.Series(self.lane_.reindex(lane).to_numpy(), index=X.index)
        pred = pred.fillna(pd.Series(self.band_.reindex(band).to_numpy(), index=X.index))
        pred = pred.fillna(keys["equipment"].map(self.equipment_).astype(float))
        return pred.fillna(self.overall_).to_numpy()

    def _keys(self, X: pd.DataFrame) -> pd.DataFrame:
        miles = np.exp(X["log_distance"])
        return pd.DataFrame({
            "pickup": X["pickup"], "delivery": X["delivery"], "equipment": X["equipment"].astype(str),
            "band": pd.cut(miles, self.BANDS, labels=False),
        }, index=X.index)
