"""Column names and paths shared by training, tests and the Streamlit app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "StudentsPrepared.xlsx"
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
FIGURES_DIR = ROOT / "docs" / "figures"

TARGET = "Target"
DROPOUT_LABEL = "Desistente"

NATIONALITY = "Nacionalidade"
HOME_NATIONALITY = "Português"
INTERNATIONAL = "International"

SEMESTERS = (1, 2)


def semester_column(semester: int, suffix: str) -> str:
    """Return the dataset column for one curricular-unit counter of a semester."""
    return f"UnidadesCurriculares{semester}Semestre{suffix}"


GRADE_COLUMNS = [semester_column(semester, "Grau") for semester in SEMESTERS]

CATEGORICAL_COLUMNS = [
    "EstadoCivil",
    "Curso",
    "QualificacaoAnterior",
    NATIONALITY,
    "Genero",
]

BINARY_COLUMNS = [
    "NecessidadesEspeciais",
    "Devedor",
    "MensalidadesEmDia",
    "Bolsista",
]

REDUNDANT_COLUMNS = [INTERNATIONAL]

SEMESTER_SUFFIXES = ["Creditado", "Inscrito", "Avaliacoes", "Aprovado", "Grau", "SemAvaliacoes"]

NUMERIC_COLUMNS = [
    "QualificacaoAnteriorGrau",
    "NotaAdmissao",
    *[semester_column(semester, suffix) for semester in SEMESTERS for suffix in SEMESTER_SUFFIXES],
    "TaxaDesemprego",
    "TaxaInflacao",
    "PIB",
]

DERIVED_COLUMNS = [
    "TaxaAprovacao1Semestre",
    "TaxaAprovacao2Semestre",
    "AvaliacoesPorUnidade1Semestre",
    "AvaliacoesPorUnidade2Semestre",
    "VariacaoNotaEntreSemestres",
    "VariacaoAprovadasEntreSemestres",
    "SemUnidadesInscritas",
]

RAW_FEATURE_COLUMNS = [
    "EstadoCivil",
    "Curso",
    "QualificacaoAnterior",
    "QualificacaoAnteriorGrau",
    NATIONALITY,
    "NotaAdmissao",
    "NecessidadesEspeciais",
    "Devedor",
    "MensalidadesEmDia",
    "Genero",
    "Bolsista",
    INTERNATIONAL,
    *[semester_column(semester, suffix) for semester in SEMESTERS for suffix in SEMESTER_SUFFIXES],
    "TaxaDesemprego",
    "TaxaInflacao",
    "PIB",
]
