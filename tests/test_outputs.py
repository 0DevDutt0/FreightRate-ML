"""Checks the real deliverables with the provided scorer's own validators.

Skipped until `python -m freight_rate predict` has written the files.
"""

import importlib.util

import pandas as pd
import pytest

from freight_rate.config import DECEMBER_PATH, PREDICTIONS_PATH, ROOT, TEMPLATE_PATH


def load_scorer():
    spec = importlib.util.spec_from_file_location("score", ROOT / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


needs_predictions = pytest.mark.skipif(not PREDICTIONS_PATH.exists(), reason="run `python -m freight_rate predict` first")


@needs_predictions
def test_validation_predictions_pass_the_official_checks():
    load_scorer().validate_predictions(pd.read_csv(PREDICTIONS_PATH))


@needs_predictions
def test_validation_predictions_keep_template_order():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    assert predictions["load_id"].tolist() == template["load_id"].tolist()


@needs_predictions
def test_december_predictions_pass_the_official_checks():
    december = load_scorer().validate_december(pd.read_csv(DECEMBER_PATH))
    assert december["predicted_rate"].between(300, 3000).all()
