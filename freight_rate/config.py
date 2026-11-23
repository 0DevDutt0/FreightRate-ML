"""Paths, constants and model settings shared by the whole pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "freight_rate_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
SCORER_CHART_PATH = ROOT / "scorer_results" / "candidate_december.png"
REPORT_PATH = REPORTS_DIR / "Freight_Rate_Model_Report.docx"

SEED = 42
EQUIPMENT_TYPES = ("Dry Van", "Flatbed", "Reefer")

# Day zero for the elapsed-time feature (first day of the development data).
TIME_ORIGIN = pd.Timestamp("2025-01-01")
DAYS_PER_MONTH = 30.44

# A label is treated as corrupted when |log(actual / expected rate)| exceeds this.
# Clean rows sit within 0.2 of the expectation and corrupted rows beyond 0.79
# (factors of roughly x0.18-0.45 and x2.2-5.5), so the cut is not sensitive.
OUTLIER_LOG_THRESHOLD = 0.35

# Main regressor. Within-period CV moved by less than 0.03 MAPE points across
# 31-127 leaves and 700-3000 trees, so a moderate setting is used.
LGBM_PARAMS: dict = {
    "objective": "regression",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbose": -1,
}

# Robust (L1) model used only to score label plausibility when hunting outliers.
OUTLIER_LGBM_PARAMS: dict = {
    **LGBM_PARAMS,
    "objective": "l1",
    "n_estimators": 600,
    "learning_rate": 0.05,
}
