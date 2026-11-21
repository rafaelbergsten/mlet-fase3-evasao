"""Preprocessing and candidate estimators for the dropout classifier."""

from dataclasses import dataclass, field

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from evasao import schema
from evasao.features import AcademicFeatures

RANDOM_STATE = 42
MIN_CATEGORY_FREQUENCY = 20


@dataclass(frozen=True)
class Candidate:
    """An estimator family with the hyperparameter grid searched for it."""

    key: str
    label: str
    estimator: object
    param_grid: dict = field(default_factory=dict)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categorical ones, keep binary flags."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, [*schema.NUMERIC_COLUMNS, *schema.DERIVED_COLUMNS]),
            ("binary", SimpleImputer(strategy="most_frequent"), schema.BINARY_COLUMNS),
            ("categorical", categorical, schema.CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Chain feature engineering, preprocessing and the classifier."""
    return Pipeline(
        [
            ("features", AcademicFeatures()),
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def candidates() -> list[Candidate]:
    """Return the estimator families compared during cross-validation."""
    return [
        Candidate(
            key="decision_stump",
            label="Árvore rasa (profundidade 1)",
            estimator=DecisionTreeClassifier(
                max_depth=1, class_weight="balanced", random_state=RANDOM_STATE
            ),
        ),
        Candidate(
            key="unpruned_tree",
            label="Árvore sem limite de profundidade",
            estimator=DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        ),
        Candidate(
            key="logistic_regression",
            label="Regressão logística",
            estimator=LogisticRegression(
                class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE
            ),
            param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        Candidate(
            key="random_forest",
            label="Random forest",
            estimator=RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
            ),
            param_grid={
                "classifier__max_depth": [None, 8, 12],
                "classifier__min_samples_leaf": [1, 5, 10],
            },
        ),
        Candidate(
            key="gradient_boosting",
            label="Gradient boosting",
            estimator=HistGradientBoostingClassifier(
                class_weight="balanced", early_stopping=False, random_state=RANDOM_STATE
            ),
            param_grid={
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__max_iter": [100, 300],
                "classifier__max_depth": [3, 6],
            },
        ),
    ]
