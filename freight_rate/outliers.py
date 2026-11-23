"""Detection of corrupted training labels.

About 1.4% of ``posted_rate`` values sit 2.2-5.5x above or 0.18-0.45x below what the rest of
the data implies for the same lane, equipment, weight and day, with nothing in between: a
pattern of data-entry corruption rather than real market moves. Each row is scored against an
out-of-fold robust (L1) model so a row never vouches for itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from .config import OUTLIER_LGBM_PARAMS, OUTLIER_LOG_THRESHOLD, SEED
from .features import MARKET_TIME


def flag_label_outliers(X: pd.DataFrame, y: pd.Series, threshold: float = OUTLIER_LOG_THRESHOLD,
                        n_splits: int = 5) -> pd.DataFrame:
    """Return expected log rate per mile, the log residual and an ``is_outlier`` flag per row."""
    expected = np.empty(len(y))
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fit_idx, score_idx in folds.split(X):
        model = LGBMRegressor(**OUTLIER_LGBM_PARAMS)
        model.fit(X.iloc[fit_idx][MARKET_TIME], y.iloc[fit_idx])
        expected[score_idx] = model.predict(X.iloc[score_idx][MARKET_TIME])
    residual = y.to_numpy() - expected
    return pd.DataFrame(
        {"expected_log_rpm": expected, "log_residual": residual, "is_outlier": np.abs(residual) > threshold},
        index=y.index,
    )


def fit_label_checker(X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
    """Robust model fitted on (already screened) rows, used to screen labels of later rows."""
    return LGBMRegressor(**OUTLIER_LGBM_PARAMS).fit(X[MARKET_TIME], y)


def flag_with_checker(checker: LGBMRegressor, X: pd.DataFrame, y: pd.Series,
                      threshold: float = OUTLIER_LOG_THRESHOLD) -> pd.Series:
    residual = y.to_numpy() - checker.predict(X[MARKET_TIME])
    return pd.Series(np.abs(residual) > threshold, index=y.index)
