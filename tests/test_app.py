from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evasao import schema

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def risk_score(app: AppTest) -> int:
    metric = next(metric for metric in app.metric if metric.label == "Score de risco de evasão")
    return int(metric.value.rstrip("%"))


def test_app_starts_without_errors(app):
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Previsão individual",
        "Previsão em lote",
        "Desempenho do modelo",
        "Sobre",
    ]


def test_high_risk_example_is_predicted_as_dropout(app):
    app.selectbox(key="example").set_value("Exemplo de risco alto").run()
    app.button[0].click().run()

    assert not app.exception
    assert risk_score(app) > 80
    assert any("evasão" in error.value for error in app.error)


def test_low_risk_example_is_predicted_as_not_dropout(app):
    app.selectbox(key="example").set_value("Exemplo de risco baixo").run()
    app.button[0].click().run()

    assert not app.exception
    assert risk_score(app) < 20
    assert any("não evasão" in success.value for success in app.success)


def test_inconsistent_counters_raise_a_warning(app):
    app.number_input(key=schema.semester_column(1, "Inscrito")).set_value(2)
    app.number_input(key=schema.semester_column(1, "Aprovado")).set_value(5)
    app.button[0].click().run()

    assert not app.exception
    assert any("mais unidades aprovadas" in warning.value for warning in app.warning)


def test_form_changes_move_the_score(app):
    app.button[0].click().run()
    typical = risk_score(app)

    app.number_input(key=schema.semester_column(2, "Aprovado")).set_value(0)
    app.number_input(key=schema.semester_column(2, "Grau")).set_value(0.0)
    app.toggle(key="MensalidadesEmDia").set_value(False)
    app.button[0].click().run()

    assert risk_score(app) > typical
