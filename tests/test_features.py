import pandas as pd
import pytest

from evasao import schema
from evasao.features import AcademicFeatures


def student(**overrides) -> pd.DataFrame:
    row = {
        schema.semester_column(semester, suffix): 0
        for semester in schema.SEMESTERS
        for suffix in schema.SEMESTER_SUFFIXES
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_approval_rate_is_approved_over_enrolled():
    frame = student(
        UnidadesCurriculares1SemestreInscrito=6,
        UnidadesCurriculares1SemestreAprovado=3,
        UnidadesCurriculares2SemestreInscrito=5,
        UnidadesCurriculares2SemestreAprovado=5,
    )

    derived = AcademicFeatures().fit_transform(frame).iloc[0]

    assert derived["TaxaAprovacao1Semestre"] == pytest.approx(0.5)
    assert derived["TaxaAprovacao2Semestre"] == pytest.approx(1.0)
    assert derived["VariacaoAprovadasEntreSemestres"] == 2
    assert derived["SemUnidadesInscritas"] == 0


def test_student_without_enrolled_units_gets_zero_rates_and_the_flag():
    derived = AcademicFeatures().fit_transform(student()).iloc[0]

    assert derived["TaxaAprovacao1Semestre"] == 0
    assert derived["AvaliacoesPorUnidade2Semestre"] == 0
    assert derived["SemUnidadesInscritas"] == 1


def test_grade_trend_is_second_minus_first_semester():
    frame = student(UnidadesCurriculares1SemestreGrau=14.0, UnidadesCurriculares2SemestreGrau=11.5)

    derived = AcademicFeatures().fit_transform(frame).iloc[0]

    assert derived["VariacaoNotaEntreSemestres"] == pytest.approx(-2.5)


def test_transform_adds_every_derived_column_without_touching_the_input():
    frame = student()
    original_columns = list(frame.columns)

    transformed = AcademicFeatures().fit_transform(frame)

    assert list(frame.columns) == original_columns
    assert list(transformed.columns) == [*original_columns, *schema.DERIVED_COLUMNS]
