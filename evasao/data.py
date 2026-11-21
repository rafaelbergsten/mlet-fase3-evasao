"""Loading and cleaning of the student performance dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from evasao import schema

GRADE_SCALE_MAX = 20.0


def repair_grade(values: pd.Series) -> pd.Series:
    """Restore grades whose decimal separator was lost in the spreadsheet export.

    Grades live on a 0-20 scale, so any value above 20 (14345, 1.36e16) is the
    original number without its decimal point. Dividing by the power of ten that
    brings it back under 20 recovers the grade.
    """
    grades = values.astype(float)
    in_scale = grades.where(grades > GRADE_SCALE_MAX, GRADE_SCALE_MAX)
    shift = np.ceil(np.log10(in_scale / GRADE_SCALE_MAX))
    return grades / np.power(10.0, shift)


def load_raw(path: Path = schema.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the spreadsheet exactly as it was distributed."""
    return pd.read_excel(path)


def repair_grades(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the frame with every grade column back on the 0-20 scale."""
    repaired = frame.copy()
    for column in schema.GRADE_COLUMNS:
        repaired[column] = repair_grade(repaired[column])
    return repaired


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Repair the grade columns and drop exact duplicate rows."""
    return repair_grades(raw).drop_duplicates().reset_index(drop=True)


def with_international_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the International column from the nationality when it is missing."""
    if schema.INTERNATIONAL in frame.columns:
        return frame
    flagged = frame.copy()
    flagged[schema.INTERNATIONAL] = (flagged[schema.NATIONALITY] != schema.HOME_NATIONALITY).astype(
        int
    )
    return flagged


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model inputs and the binary target (1 = dropout)."""
    features = frame[schema.RAW_FEATURE_COLUMNS]
    target = (frame[schema.TARGET] == schema.DROPOUT_LABEL).astype(int).rename("Evasao")
    return features, target


def load_dataset(path: Path = schema.RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load, clean and split the dataset into features and binary target."""
    return split_features_target(clean(load_raw(path)))


def prepare_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn rows in the spreadsheet layout into the columns the model expects."""
    flagged = with_international_flag(frame)
    missing = [column for column in schema.RAW_FEATURE_COLUMNS if column not in flagged.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return repair_grades(flagged)[schema.RAW_FEATURE_COLUMNS]
