"""Feature engineering steps that travel inside the fitted pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from evasao import schema


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0)).fillna(0.0)


class AcademicFeatures(TransformerMixin, BaseEstimator):
    """Add ratio and trend features computed from the per-semester counters."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - scikit-learn API name
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803 - scikit-learn API name
        frame = X.copy()
        enrolled = {s: frame[schema.semester_column(s, "Inscrito")] for s in schema.SEMESTERS}
        approved = {s: frame[schema.semester_column(s, "Aprovado")] for s in schema.SEMESTERS}
        evaluations = {s: frame[schema.semester_column(s, "Avaliacoes")] for s in schema.SEMESTERS}
        grade = {s: frame[schema.semester_column(s, "Grau")] for s in schema.SEMESTERS}

        for semester in schema.SEMESTERS:
            frame[f"TaxaAprovacao{semester}Semestre"] = _safe_ratio(
                approved[semester], enrolled[semester]
            )
            frame[f"AvaliacoesPorUnidade{semester}Semestre"] = _safe_ratio(
                evaluations[semester], enrolled[semester]
            )
        frame["VariacaoNotaEntreSemestres"] = grade[2] - grade[1]
        frame["VariacaoAprovadasEntreSemestres"] = approved[2] - approved[1]
        frame["SemUnidadesInscritas"] = ((enrolled[1] == 0) & (enrolled[2] == 0)).astype(int)
        return frame[[*X.columns, *schema.DERIVED_COLUMNS]]

    def get_feature_names_out(self, input_features=None):
        """Return the input columns followed by the derived ones."""
        return [*list(input_features or []), *schema.DERIVED_COLUMNS]
