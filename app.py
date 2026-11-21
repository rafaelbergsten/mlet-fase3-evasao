"""Streamlit app that serves the student dropout classifier."""

import pandas as pd
import streamlit as st

from evasao import data, plots, schema
from evasao.features import AcademicFeatures
from evasao.predict import (
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    is_linear,
    linear_contributions,
    load_metadata,
    load_model,
    score,
)

TYPICAL_STUDENT = "Aluno típico (medianas da base)"
EXAMPLE_KEY = "example"
INFREQUENT_SUFFIX = "infrequent_sklearn"
TOP_FACTORS = 8

FIELD_LABELS = {
    "EstadoCivil": "Estado civil",
    "Curso": "Curso",
    "QualificacaoAnterior": "Qualificação anterior",
    "QualificacaoAnteriorGrau": "Nota da qualificação anterior",
    "Nacionalidade": "Nacionalidade",
    "NotaAdmissao": "Nota de admissão",
    "NecessidadesEspeciais": "Necessidades educacionais especiais",
    "Devedor": "Devedor",
    "MensalidadesEmDia": "Mensalidades em dia",
    "Genero": "Gênero",
    "Bolsista": "Bolsista",
    "TaxaDesemprego": "Taxa de desemprego (%)",
    "TaxaInflacao": "Taxa de inflação (%)",
    "PIB": "Variação do PIB (%)",
    "TaxaAprovacao1Semestre": "Taxa de aprovação no 1º semestre",
    "TaxaAprovacao2Semestre": "Taxa de aprovação no 2º semestre",
    "AvaliacoesPorUnidade1Semestre": "Avaliações por unidade no 1º semestre",
    "AvaliacoesPorUnidade2Semestre": "Avaliações por unidade no 2º semestre",
    "VariacaoNotaEntreSemestres": "Variação da nota entre os semestres",
    "VariacaoAprovadasEntreSemestres": "Variação de aprovadas entre os semestres",
    "SemUnidadesInscritas": "Sem unidades inscritas nos dois semestres",
}

COUNTER_LABELS = {
    "Creditado": "Unidades creditadas",
    "Inscrito": "Unidades inscritas",
    "Avaliacoes": "Avaliações realizadas",
    "Aprovado": "Unidades aprovadas",
    "Grau": "Nota média (0 a 20)",
    "SemAvaliacoes": "Unidades sem avaliação",
}

for _semester in schema.SEMESTERS:
    for _suffix, _label in COUNTER_LABELS.items():
        FIELD_LABELS[schema.semester_column(_semester, _suffix)] = (
            f"{_label} no {_semester}º semestre"
        )

FLOAT_FIELDS = {
    "QualificacaoAnteriorGrau": (0.0, 200.0, 0.1),
    "NotaAdmissao": (0.0, 200.0, 0.1),
    "TaxaDesemprego": (0.0, 30.0, 0.1),
    "TaxaInflacao": (-5.0, 15.0, 0.1),
    "PIB": (-10.0, 10.0, 0.01),
    **dict.fromkeys(schema.GRADE_COLUMNS, (0.0, 20.0, 0.1)),
}

COUNTER_MAX = 60

METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "f1": "F1",
    "recall": "Recall",
    "precision": "Precisão",
    "accuracy": "Acurácia",
}


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metadata() -> dict:
    return load_metadata()


def is_float_field(column: str) -> bool:
    return column in FLOAT_FIELDS


def typical_student(metadata: dict) -> dict:
    inputs = metadata["inputs"]
    values = {column: options[0] for column, options in inputs["categories"].items()}
    values.update({column: stats["median"] for column, stats in inputs["numeric"].items()})
    values.update({"NecessidadesEspeciais": 0, "Devedor": 0, "MensalidadesEmDia": 1, "Bolsista": 0})
    return values


def profiles(metadata: dict) -> dict[str, dict]:
    examples = {example["label"]: example["values"] for example in metadata["examples"]}
    return {TYPICAL_STUDENT: typical_student(metadata), **examples}


def fill_form(values: dict) -> None:
    for column in schema.CATEGORICAL_COLUMNS:
        st.session_state[column] = values[column]
    for column in schema.BINARY_COLUMNS:
        st.session_state[column] = bool(values[column])
    for column in schema.NUMERIC_COLUMNS:
        cast = float if is_float_field(column) else int
        st.session_state[column] = cast(values[column])


def apply_selected_example() -> None:
    fill_form(profiles(get_metadata())[st.session_state[EXAMPLE_KEY]])


def number_field(column: str, label: str | None = None) -> None:
    if is_float_field(column):
        minimum, maximum, step = FLOAT_FIELDS[column]
        st.number_input(
            label or FIELD_LABELS[column],
            min_value=minimum,
            max_value=maximum,
            step=step,
            key=column,
        )
    else:
        st.number_input(
            label or FIELD_LABELS[column], min_value=0, max_value=COUNTER_MAX, step=1, key=column
        )


def collect_form() -> pd.DataFrame:
    row = {column: st.session_state[column] for column in schema.CATEGORICAL_COLUMNS}
    row.update({column: int(st.session_state[column]) for column in schema.BINARY_COLUMNS})
    row.update({column: st.session_state[column] for column in schema.NUMERIC_COLUMNS})
    return data.with_international_flag(pd.DataFrame([row]))[schema.RAW_FEATURE_COLUMNS]


def consistency_warnings(row: pd.Series) -> list[str]:
    warnings = []
    for semester in schema.SEMESTERS:
        enrolled = row[schema.semester_column(semester, "Inscrito")]
        approved = row[schema.semester_column(semester, "Aprovado")]
        grade = row[schema.semester_column(semester, "Grau")]
        if approved > enrolled:
            warnings.append(
                f"{semester}º semestre: há mais unidades aprovadas ({approved}) do que inscritas ({enrolled})."
            )
        if approved == 0 and grade > 0:
            warnings.append(
                f"{semester}º semestre: na base, quem não tem unidade aprovada tem nota média 0."
            )
        if approved > 0 and grade == 0:
            warnings.append(
                f"{semester}º semestre: há unidades aprovadas, mas a nota média está em 0."
            )
    return warnings


def humanize_feature(name: str) -> str:
    for column in schema.CATEGORICAL_COLUMNS:
        prefix = f"{column}_"
        if name.startswith(prefix):
            category = name.removeprefix(prefix)
            category = "categorias raras" if category == INFREQUENT_SUFFIX else category
            return f"{FIELD_LABELS[column]}: {category}"
    return FIELD_LABELS.get(name, name)


def risk_band(probability: float, threshold: float) -> str:
    if probability >= max(threshold, 0.7):
        return "alto"
    if probability >= threshold:
        return "moderado"
    return "baixo"


def contributions_chart(contributions: pd.DataFrame):
    top = contributions.head(TOP_FACTORS).iloc[::-1]
    colors = [
        plots.VALIDATION_COLOR if value > 0 else plots.TRAIN_COLOR for value in top["contribution"]
    ]
    fig, ax = plots.plt.subplots(figsize=(7.5, 0.42 * len(top) + 1.1))
    ax.barh(
        [humanize_feature(name) for name in top["feature"]],
        top["contribution"],
        color=colors,
        height=0.6,
    )
    ax.axvline(0, color=plots.TEXT_SECONDARY, linewidth=1)
    ax.set_xlabel("Contribuição no log-odds (direita aumenta o risco, esquerda reduz)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def render_prediction(model, metadata: dict, inputs: pd.DataFrame) -> None:
    threshold = metadata["decision_threshold"]
    result = score(model, inputs, threshold).iloc[0]
    probability = float(result[PROBABILITY_COLUMN])
    band = risk_band(probability, threshold)

    for warning in consistency_warnings(inputs.iloc[0]):
        st.warning(warning)

    derived = AcademicFeatures().fit_transform(inputs).iloc[0]
    left, middle, right = st.columns(3)
    left.metric("Score de risco de evasão", f"{probability:.0%}")
    middle.metric("Aprovação no 1º semestre", f"{derived['TaxaAprovacao1Semestre']:.0%}")
    right.metric("Aprovação no 2º semestre", f"{derived['TaxaAprovacao2Semestre']:.0%}")
    st.progress(probability)

    if result[PREDICTION_COLUMN] == 1:
        st.error(
            f"Previsão: **evasão** (risco {band}). O score ficou acima do ponto de corte de {threshold:.0%}."
        )
    else:
        st.success(
            f"Previsão: **não evasão** (risco {band}). O score ficou abaixo do ponto de corte de {threshold:.0%}."
        )

    if is_linear(model):
        st.subheader("O que mais pesou nesta previsão")
        st.caption(
            "Cada barra compara este aluno com o aluno médio do treino. "
            "Laranja empurra o score para cima; azul puxa para baixo."
        )
        contributions = linear_contributions(model, inputs, metadata["transformed_feature_means"])
        chart, _ = st.columns([3, 1])
        chart.pyplot(contributions_chart(contributions))


def render_individual_tab(model, metadata: dict) -> None:
    st.selectbox(
        "Preencher o formulário com",
        list(profiles(metadata)),
        key=EXAMPLE_KEY,
        on_change=apply_selected_example,
        help="Os exemplos são alunos do conjunto de teste, que o modelo não viu no treino.",
    )
    categories = metadata["inputs"]["categories"]

    with st.form("student"):
        st.subheader("Perfil")
        profile = st.columns(4)
        for column, name in zip(
            profile, ["Curso", "Genero", "EstadoCivil", "Nacionalidade"], strict=True
        ):
            column.selectbox(FIELD_LABELS[name], categories[name], key=name)
        background = st.columns(3)
        background[0].selectbox(
            FIELD_LABELS["QualificacaoAnterior"],
            categories["QualificacaoAnterior"],
            key="QualificacaoAnterior",
        )
        with background[1]:
            number_field("QualificacaoAnteriorGrau")
        with background[2]:
            number_field("NotaAdmissao")

        st.subheader("Situação financeira e apoio")
        toggles = st.columns(4)
        for column, name in zip(toggles, schema.BINARY_COLUMNS, strict=True):
            column.toggle(FIELD_LABELS[name], key=name)

        for semester in schema.SEMESTERS:
            st.subheader(f"{semester}º semestre")
            columns = st.columns(len(COUNTER_LABELS))
            for column, (suffix, label) in zip(columns, COUNTER_LABELS.items(), strict=True):
                with column:
                    number_field(schema.semester_column(semester, suffix), label)

        st.subheader("Contexto econômico no ano da matrícula")
        macro = st.columns(3)
        for column, name in zip(macro, ["TaxaDesemprego", "TaxaInflacao", "PIB"], strict=True):
            with column:
                number_field(name)

        submitted = st.form_submit_button("Calcular risco de evasão", type="primary")

    if submitted:
        render_prediction(model, metadata, collect_form())


def render_batch_tab(model, metadata: dict) -> None:
    st.write(
        "Envie uma planilha no mesmo formato da base (`.xlsx` ou `.csv`). "
        "As notas com o separador decimal perdido são corrigidas antes da previsão."
    )
    uploaded = st.file_uploader("Planilha de alunos", type=["xlsx", "csv"])
    if uploaded is None:
        return

    frame = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    try:
        inputs = data.prepare_inputs(frame)
    except ValueError as error:
        missing = str(error).split(":", 1)[-1].strip()
        st.error(f"A planilha não tem todas as colunas esperadas. Faltam: {missing}")
        return

    extra = frame.drop(columns=inputs.columns, errors="ignore")
    result = pd.concat(
        [score(model, inputs, metadata["decision_threshold"]), inputs, extra], axis=1
    )
    flagged = int(result[PREDICTION_COLUMN].sum())
    left, middle, right = st.columns(3)
    left.metric("Alunos avaliados", len(result))
    middle.metric("Previstos como evasão", flagged)
    right.metric("Proporção em risco", f"{flagged / len(result):.0%}")

    st.dataframe(
        result.sort_values(PROBABILITY_COLUMN, ascending=False),
        hide_index=True,
        column_config={
            PROBABILITY_COLUMN: st.column_config.ProgressColumn(
                "Score de risco", min_value=0.0, max_value=1.0, format="percent"
            ),
            PREDICTION_COLUMN: st.column_config.NumberColumn("Previsão (1 = evasão)"),
        },
    )
    st.download_button(
        "Baixar previsões em CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name="previsoes_evasao.csv",
        mime="text/csv",
    )


def cross_validation_table(metadata: dict) -> pd.DataFrame:
    rows = []
    for row in metadata["cross_validation"]:
        rows.append(
            {
                "Modelo": row["label"],
                "AUC treino": row["train_roc_auc"],
                "AUC validação": row["validation_roc_auc"],
                "Diferença": row["gap"],
                "F1 treino": row["train_f1"],
                "F1 validação": row["validation_f1"],
                "Recall validação": row["validation_recall"],
            }
        )
    return pd.DataFrame(rows)


def figure(name: str, caption: str) -> None:
    st.image(str(schema.FIGURES_DIR / name), caption=caption)


def render_performance_tab(metadata: dict) -> None:
    test = metadata["test"]
    rows = metadata["rows"]
    selected = metadata["selected"]
    by_key = {row["key"]: row for row in metadata["cross_validation"]}
    chosen = by_key[selected["key"]]
    stump = by_key["decision_stump"]
    unpruned = by_key["unpruned_tree"]

    st.subheader(f"Modelo em produção: {selected['label']}")
    st.caption(
        f"Treinado com {rows['train']} alunos e avaliado uma única vez em {rows['test']} alunos "
        f"separados antes de qualquer ajuste. Ponto de corte: {metadata['decision_threshold']:.0%}."
    )
    for column, (metric, label) in zip(
        st.columns(len(METRIC_LABELS)), METRIC_LABELS.items(), strict=True
    ):
        column.metric(f"{label} no teste", f"{test[metric]:.3f}")

    st.subheader("Validação cruzada: treino x validação")
    st.write(
        f"Cada modelo passou por validação cruzada estratificada de {metadata['cv_folds']} folds "
        "dentro do conjunto de treino. A coluna **Diferença** é o AUC de treino menos o de validação."
    )
    st.dataframe(cross_validation_table(metadata).style.format(precision=3), hide_index=True)
    figure("cv_roc_auc.png", "ROC AUC médio em treino e validação para cada candidato.")

    st.subheader("Houve overfitting ou underfitting?")
    st.markdown(
        f"""
- **Underfitting na árvore rasa.** Com uma única divisão, o AUC fica em {stump["train_roc_auc"]:.3f} no treino
  e {stump["validation_roc_auc"]:.3f} na validação. Os dois são baixos e quase iguais: o modelo é simples demais
  para os dados.
- **Overfitting na árvore sem limite.** O AUC chega a {unpruned["train_roc_auc"]:.3f} no treino e cai para
  {unpruned["validation_roc_auc"]:.3f} na validação. A árvore decorou o treino.
- **Ajuste adequado no modelo escolhido.** A {selected["label"].lower()} tem AUC de
  {chosen["train_roc_auc"]:.3f} no treino, {chosen["validation_roc_auc"]:.3f} na validação e
  {test["roc_auc"]:.3f} no teste. A diferença entre treino e validação é de {chosen["gap"]:.3f}.
"""
    )
    figure(
        "validation_curve.png",
        "A mesma árvore de decisão com profundidades de 1 a 20: a validação sobe, atinge o pico e depois cai.",
    )
    figure(
        "learning_curve.png",
        "Com mais alunos no treino, as curvas de treino e validação se aproximam.",
    )

    st.subheader("Resultado no conjunto de teste")
    left, right = st.columns(2)
    with left:
        figure("confusion_matrix.png", "Matriz de confusão no teste.")
    with right:
        figure("roc_curve.png", "Curva ROC no teste.")
    figure(
        "feature_importance.png",
        "Importância por permutação: quanto o AUC cai quando cada coluna é embaralhada.",
    )


def render_about_tab(metadata: dict) -> None:
    rows = metadata["rows"]
    st.markdown(
        f"""
### O problema
Prever quais alunos de uma faculdade vão abandonar o curso, a partir do perfil, da situação financeira
e do desempenho nos dois primeiros semestres.

### A base
- {rows["raw"]} alunos e 27 variáveis explicativas; {rows["clean"]} alunos depois de remover uma linha duplicada.
- A coluna `Target` tem três situações: Graduado, Matriculado e Desistente. O alvo binário é
  **Desistente = 1** e as outras duas = 0. A evasão representa {metadata["dropout_rate"]:.0%} da base.
- As notas médias por semestre vieram sem o separador decimal em parte das linhas
  (`13,67` aparecia como `1,36e16`). Elas foram trazidas de volta para a escala de 0 a 20.

### A pipeline
1. **Features derivadas**: taxa de aprovação e avaliações por unidade em cada semestre, variação da nota e
   das aprovações entre os semestres, e um indicador de aluno sem unidades inscritas.
2. **Numéricas**: imputação pela mediana e padronização.
3. **Categóricas**: one-hot encoding, com as categorias raras agrupadas.
4. **Modelo**: {metadata["selected"]["label"].lower()}, escolhida por validação cruzada.

### Limites
O score vem de um modelo treinado com pesos balanceados entre as classes, então ele ordena bem os alunos
por risco, mas não deve ser lido como probabilidade calibrada. As variáveis de desempenho só existem depois
do 2º semestre: o modelo não serve para prever evasão no momento da matrícula.
"""
    )


def main() -> None:
    st.set_page_config(
        page_title="Previsão de evasão", page_icon=":material/school:", layout="wide"
    )
    model = get_model()
    metadata = get_metadata()

    if "Curso" not in st.session_state:
        fill_form(typical_student(metadata))

    st.title("Previsão de evasão de estudantes")
    st.caption(
        "Prova substitutiva da Fase 3 de Machine Learning Engineering. "
        "Modelo binário treinado com dados de desempenho acadêmico."
    )

    individual, batch, performance, about = st.tabs(
        ["Previsão individual", "Previsão em lote", "Desempenho do modelo", "Sobre"]
    )
    with individual:
        render_individual_tab(model, metadata)
    with batch:
        render_batch_tab(model, metadata)
    with performance:
        render_performance_tab(metadata)
    with about:
        render_about_tab(metadata)


main()
