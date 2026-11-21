import pandas as pd
import pytest

from evasao import data, schema


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (0.0, 0.0),
        (13.3, 13.3),
        (20.0, 20.0),
        (14345.0, 14.345),
        (13875.0, 13.875),
        (1.36666666666666e16, 13.6666666666666),
        (1.34285714285714e16, 13.4285714285714),
    ],
)
def test_repair_grade_restores_the_lost_decimal_separator(stored, expected):
    repaired = data.repair_grade(pd.Series([stored]))

    assert repaired.iloc[0] == pytest.approx(expected)


def test_clean_brings_every_grade_back_to_the_scale(raw):
    frame = data.clean(raw)

    for column in schema.GRADE_COLUMNS:
        assert frame[column].between(0, data.GRADE_SCALE_MAX).all()


def test_repaired_grade_is_zero_exactly_when_no_unit_was_approved(raw):
    frame = data.clean(raw)

    for semester in schema.SEMESTERS:
        no_grade = frame[schema.semester_column(semester, "Grau")] == 0
        no_approval = frame[schema.semester_column(semester, "Aprovado")] == 0
        assert (no_grade == no_approval).all()


def test_clean_drops_exact_duplicates(raw):
    frame = data.clean(raw)

    assert len(frame) == len(raw) - 1
    assert not frame.duplicated().any()


def test_target_is_one_only_for_dropouts(raw):
    frame = data.clean(raw)
    _, target = data.split_features_target(frame)

    assert set(target.unique()) == {0, 1}
    assert target.sum() == (frame[schema.TARGET] == schema.DROPOUT_LABEL).sum()


def test_features_do_not_carry_the_target(raw):
    features, _ = data.split_features_target(data.clean(raw))

    assert list(features.columns) == schema.RAW_FEATURE_COLUMNS
    assert schema.TARGET not in features.columns


def test_international_flag_is_derived_from_nationality():
    frame = pd.DataFrame({schema.NATIONALITY: [schema.HOME_NATIONALITY, "Brasileiro"]})

    flagged = data.with_international_flag(frame)

    assert flagged[schema.INTERNATIONAL].tolist() == [0, 1]


def test_international_flag_matches_the_column_shipped_with_the_dataset(raw):
    derived = data.with_international_flag(raw.drop(columns=schema.INTERNATIONAL))

    assert (derived[schema.INTERNATIONAL] == raw[schema.INTERNATIONAL]).all()


def test_prepare_inputs_reports_missing_columns(raw):
    with pytest.raises(ValueError, match="Curso"):
        data.prepare_inputs(raw.drop(columns="Curso"))
