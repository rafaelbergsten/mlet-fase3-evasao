"""Inference helpers shared by the Streamlit app and the tests."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evasao import schema

PROBABILITY_COLUMN = "ProbabilidadeEvasao"
PREDICTION_COLUMN = "PrevisaoEvasao"


def load_model(path: Path = schema.MODEL_PATH):
    """Load the fitted pipeline from disk."""
    return joblib.load(path)


def load_metadata(path: Path = schema.METADATA_PATH) -> dict:
    """Load the training report written next to the model."""
    return json.loads(path.read_text())


def score(model, inputs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return the dropout probability and the binary prediction for each row."""
    probability = model.predict_proba(inputs[schema.RAW_FEATURE_COLUMNS])[:, 1]
    return pd.DataFrame(
        {
            PROBABILITY_COLUMN: probability,
            PREDICTION_COLUMN: (probability >= threshold).astype(int),
        },
        index=inputs.index,
    )


def is_linear(model) -> bool:
    """Tell whether the classifier exposes per-feature coefficients."""
    return hasattr(model.named_steps["classifier"], "coef_")


def linear_contributions(model, inputs: pd.DataFrame, reference: list[float]) -> pd.DataFrame:
    """Break one prediction of a linear model into per-feature log-odds contributions.

    Each contribution is the coefficient times the distance between the student
    and the average training student, so positive values raise the dropout risk.
    """
    transformed = model[:-1].transform(inputs[schema.RAW_FEATURE_COLUMNS])
    coefficients = model.named_steps["classifier"].coef_[0]
    contribution = coefficients * (np.asarray(transformed)[0] - np.asarray(reference))
    return (
        pd.DataFrame(
            {
                "feature": model.named_steps["preprocessor"].get_feature_names_out(),
                "contribution": contribution,
            }
        )
        .sort_values("contribution", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )
