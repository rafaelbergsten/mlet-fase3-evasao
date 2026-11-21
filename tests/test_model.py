import io
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from evasao import data, predict, schema, train
from evasao.model import build_pipeline

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def pinned_versions() -> dict[str, str]:
    lines = [line.strip() for line in REQUIREMENTS.read_text().splitlines() if "==" in line]
    return dict(line.split("==", 1) for line in lines)


def test_pipeline_survives_categories_missing_from_a_training_fold(raw):
    features, target = data.split_features_target(data.clean(raw))
    rare = features["Nacionalidade"] == "Lituano"
    pipeline = build_pipeline(LogisticRegression(max_iter=2000))

    pipeline.fit(features[~rare], target[~rare])

    assert pipeline.predict_proba(features[rare]).shape == (int(rare.sum()), 2)


def test_selection_prefers_the_smallest_gap_among_statistical_ties():
    results = pd.DataFrame(
        {
            "key": ["complex", "simple", "weak"],
            "validation_roc_auc": [0.915, 0.912, 0.810],
            "validation_roc_auc_std": [0.010, 0.010, 0.010],
            "gap": [0.040, 0.008, 0.002],
        }
    )

    assert train.select(results) == "simple"


def test_split_is_stratified_and_disjoint(raw):
    features, target = data.split_features_target(data.clean(raw))

    x_train, x_test, y_train, y_test = train.split(features, target)

    assert set(x_train.index).isdisjoint(x_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_saved_model_scores_raw_rows(model, metadata, raw):
    inputs = data.prepare_inputs(raw.head(50))

    result = predict.score(model, inputs, metadata["decision_threshold"])

    assert result[predict.PROBABILITY_COLUMN].between(0, 1).all()
    assert set(result[predict.PREDICTION_COLUMN].unique()) <= {0, 1}


def test_saved_model_accepts_an_unseen_category(model, metadata, raw):
    inputs = data.prepare_inputs(raw.head(1)).assign(Curso="Curso que não existe na base")

    result = predict.score(model, inputs, metadata["decision_threshold"])

    assert 0 <= result[predict.PROBABILITY_COLUMN].iloc[0] <= 1


def test_saved_model_scores_a_csv_export_with_blank_cells(model, metadata, raw):
    sample = raw.head(20).astype({"NotaAdmissao": float, "Bolsista": float}).copy()
    sample.loc[0, ["Curso", "NotaAdmissao", "Bolsista"]] = [None, None, None]
    sample.loc[1, "UnidadesCurriculares2SemestreGrau"] = None
    exported = io.StringIO(sample.to_csv(index=False))

    inputs = data.prepare_inputs(pd.read_csv(exported))
    result = predict.score(model, inputs, metadata["decision_threshold"])

    assert inputs.isna().sum().sum() == 4
    assert result[predict.PROBABILITY_COLUMN].between(0, 1).all()


def test_saved_model_reproduces_the_reported_test_metrics(model, metadata, raw):
    features, target = data.split_features_target(data.clean(raw))
    _, x_test, _, y_test = train.split(features, target)

    metrics = train.evaluate(model, x_test, y_test)

    assert metrics == metadata["test"]


def test_contributions_add_up_to_the_log_odds_distance_from_the_average_student(
    model, metadata, raw
):
    inputs = data.prepare_inputs(raw.head(1))
    reference = np.asarray(metadata["transformed_feature_means"])
    classifier = model.named_steps["classifier"]

    contributions = predict.linear_contributions(model, inputs, reference.tolist())

    student_log_odds = model.decision_function(inputs)[0]
    average_log_odds = classifier.decision_function(reference.reshape(1, -1))[0]
    assert np.isclose(contributions["contribution"].sum(), student_log_odds - average_log_odds)


def test_examples_cover_low_and_high_risk(model, metadata):
    examples = pd.DataFrame([example["values"] for example in metadata["examples"]])

    probability = predict.score(model, examples, metadata["decision_threshold"])[
        predict.PROBABILITY_COLUMN
    ]

    assert probability.iloc[0] < 0.2
    assert probability.iloc[-1] > 0.8


def test_requirements_pin_the_version_that_trained_the_model(metadata):
    pinned = pinned_versions()

    assert pinned["scikit-learn"] == metadata["scikit_learn_version"] == sklearn.__version__
    assert set(pinned) >= {"streamlit", "pandas", "joblib", "openpyxl", "matplotlib", "numpy"}


def test_every_figure_shown_by_the_app_exists():
    expected = {
        "cv_roc_auc.png",
        "validation_curve.png",
        "learning_curve.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "feature_importance.png",
    }

    assert expected <= {path.name for path in schema.FIGURES_DIR.glob("*.png")}
