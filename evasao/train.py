"""Train, select and evaluate the dropout classifier, then persist the artifacts."""

import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.tree import DecisionTreeClassifier

from evasao import data, plots, schema
from evasao.model import RANDOM_STATE, Candidate, build_pipeline, candidates

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
N_SPLITS = 5
DECISION_THRESHOLD = 0.5
SELECTION_METRIC = "roc_auc"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
TREE_DEPTHS = list(range(1, 21))


def split(features: pd.DataFrame, target: pd.Series):
    """Hold out a stratified test set that is only used for the final evaluation."""
    return train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE
    )


def cross_validator() -> StratifiedKFold:
    """Return the stratified k-fold splitter shared by every experiment."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def search(candidate: Candidate, x_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Grid-search one candidate with cross-validation, keeping the train scores."""
    grid = GridSearchCV(
        build_pipeline(candidate.estimator),
        candidate.param_grid,
        scoring=SCORING,
        refit=SELECTION_METRIC,
        cv=cross_validator(),
        return_train_score=True,
        n_jobs=-1,
    )
    return grid.fit(x_train, y_train)


def summarize(candidate: Candidate, grid: GridSearchCV) -> dict:
    """Collect mean train and validation scores of the best configuration."""
    best = grid.best_index_
    row = {
        "key": candidate.key,
        "label": candidate.label,
        "params": {name.split("__", 1)[1]: value for name, value in grid.best_params_.items()},
    }
    for metric in SCORING:
        row[f"train_{metric}"] = float(grid.cv_results_[f"mean_train_{metric}"][best])
        row[f"validation_{metric}"] = float(grid.cv_results_[f"mean_test_{metric}"][best])
        row[f"validation_{metric}_std"] = float(grid.cv_results_[f"std_test_{metric}"][best])
    row["gap"] = row[f"train_{SELECTION_METRIC}"] - row[f"validation_{SELECTION_METRIC}"]
    return row


def select(results: pd.DataFrame) -> str:
    """Pick the candidate with the smallest train-validation gap among the statistical ties.

    A candidate ties with the best one when its mean validation score is within
    one standard error of the best mean (the one-standard-error rule).
    """
    metric = f"validation_{SELECTION_METRIC}"
    best = results.loc[results[metric].idxmax()]
    standard_error = best[f"{metric}_std"] / np.sqrt(N_SPLITS)
    tied = results[results[metric] >= best[metric] - standard_error]
    return tied.loc[tied["gap"].idxmin(), "key"]


def evaluate(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score the fitted model on the held-out test set."""
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= DECISION_THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted)),
        "recall": float(recall_score(y_test, predicted)),
        "f1": float(f1_score(y_test, predicted)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
    }


def tree_depth_curve(x_train: pd.DataFrame, y_train: pd.Series):
    """Cross-validated score of a decision tree for each maximum depth."""
    tree = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    return validation_curve(
        build_pipeline(tree),
        x_train,
        y_train,
        param_name="classifier__max_depth",
        param_range=TREE_DEPTHS,
        scoring=SELECTION_METRIC,
        cv=cross_validator(),
        n_jobs=-1,
    )


def raw_importances(model, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Permutation importance of each raw input column, measured in ROC AUC."""
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring=SELECTION_METRIC,
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def example_students(model, x_test: pd.DataFrame) -> list[dict]:
    """Pick three test students spread across the predicted risk range."""
    probability = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index).sort_values()
    picks = {
        "Exemplo de risco baixo": probability.index[int(len(probability) * 0.05)],
        "Exemplo de risco intermediário": (probability - 0.5).abs().idxmin(),
        "Exemplo de risco alto": probability.index[int(len(probability) * 0.95)],
    }
    return [
        {"label": label, "values": json.loads(x_test.loc[[index]].to_json(orient="records"))[0]}
        for label, index in picks.items()
    ]


def input_options(features: pd.DataFrame) -> dict:
    """Describe the observed categories and numeric ranges for the app form."""
    categories = {
        column: features[column].value_counts().index.tolist()
        for column in schema.CATEGORICAL_COLUMNS
    }
    numeric = {
        column: {
            "min": float(features[column].min()),
            "max": float(features[column].max()),
            "median": float(features[column].median()),
        }
        for column in schema.NUMERIC_COLUMNS
    }
    return {"categories": categories, "numeric": numeric}


def run() -> dict:
    """Execute the whole training workflow and write model, metadata and figures."""
    raw = data.load_raw()
    frame = data.clean(raw)
    features, target = data.split_features_target(frame)
    x_train, x_test, y_train, y_test = split(features, target)
    logger.info(
        "Dataset: %d rows (%d raw), dropout rate %.3f, train %d, test %d",
        len(frame),
        len(raw),
        target.mean(),
        len(x_train),
        len(x_test),
    )

    searches = {}
    rows = []
    for candidate in candidates():
        searches[candidate.key] = search(candidate, x_train, y_train)
        rows.append(summarize(candidate, searches[candidate.key]))
        logger.info(
            "%s: train AUC %.4f, validation AUC %.4f, params %s",
            candidate.key,
            rows[-1]["train_roc_auc"],
            rows[-1]["validation_roc_auc"],
            rows[-1]["params"],
        )
    results = pd.DataFrame(rows)
    selected_key = select(results)
    selected = results.set_index("key").loc[selected_key]
    model = searches[selected_key].best_estimator_
    logger.info("Selected model: %s", selected_key)

    test_metrics = evaluate(model, x_test, y_test)
    logger.info("Test metrics: %s", test_metrics)

    depth_train, depth_validation = tree_depth_curve(x_train, y_train)
    sizes, curve_train, curve_validation = learning_curve(
        model,
        x_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=SELECTION_METRIC,
        cv=cross_validator(),
        n_jobs=-1,
    )
    importances = raw_importances(model, x_test, y_test)
    probability = model.predict_proba(x_test)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)

    figures = schema.FIGURES_DIR
    grade_column = schema.GRADE_COLUMNS[0]
    plots.target_distribution(
        frame[schema.TARGET], schema.DROPOUT_LABEL, figures / "target_distribution.png"
    )
    plots.grade_repair(raw[grade_column], frame[grade_column], figures / "grade_repair.png")
    plots.train_vs_validation(results, "roc_auc", "ROC AUC", figures / "cv_roc_auc.png")
    plots.train_vs_validation(results, "f1", "F1", figures / "cv_f1.png")
    plots.validation_curve_figure(
        TREE_DEPTHS, depth_train, depth_validation, "ROC AUC", figures / "validation_curve.png"
    )
    plots.learning_curve_figure(
        sizes,
        curve_train,
        curve_validation,
        "ROC AUC",
        selected["label"],
        figures / "learning_curve.png",
    )
    plots.confusion_matrix_figure(
        np.array(test_metrics["confusion_matrix"]), figures / "confusion_matrix.png"
    )
    plots.roc_curve_figure(
        false_positive_rate, true_positive_rate, test_metrics["roc_auc"], figures / "roc_curve.png"
    )
    plots.feature_importance_figure(importances, figures / "feature_importance.png")

    metadata = {
        "scikit_learn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "decision_threshold": DECISION_THRESHOLD,
        "selection_metric": SELECTION_METRIC,
        "rows": {
            "raw": len(raw),
            "clean": len(frame),
            "train": len(x_train),
            "test": len(x_test),
        },
        "dropout_rate": float(target.mean()),
        "cv_folds": N_SPLITS,
        "selected": {"key": selected_key, "label": selected["label"], "params": selected["params"]},
        "cross_validation": rows,
        "test": test_metrics,
        "tree_depth_curve": {
            "depths": TREE_DEPTHS,
            "train": depth_train.mean(axis=1).tolist(),
            "validation": depth_validation.mean(axis=1).tolist(),
        },
        "learning_curve": {
            "sizes": sizes.tolist(),
            "train": curve_train.mean(axis=1).tolist(),
            "validation": curve_validation.mean(axis=1).tolist(),
        },
        "importances": importances.to_dict(orient="records"),
        "inputs": input_options(features),
        "transformed_feature_means": np.asarray(model[:-1].transform(x_train))
        .mean(axis=0)
        .tolist(),
        "examples": example_students(model, x_test),
    }

    schema.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, schema.MODEL_PATH)
    schema.METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    logger.info("Saved %s and %s", schema.MODEL_PATH.name, schema.METADATA_PATH.name)
    return metadata


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing")
    run()


if __name__ == "__main__":
    main()
