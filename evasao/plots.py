"""Static figures used by the README, the notebook and the Streamlit app."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TRAIN_COLOR = "#2a78d6"
VALIDATION_COLOR = "#eb6834"
NEUTRAL_COLOR = "#a9a8a2"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#2a78d6", "#0d366b"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 5,
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def target_distribution(original_target: pd.Series, dropout_label: str, path: Path) -> Path:
    """Bar chart of the three original classes, split by the binary target."""
    counts = original_target.value_counts()
    colors = [VALIDATION_COLOR if label == dropout_label else TRAIN_COLOR for label in counts.index]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(counts.index, counts.to_numpy(), color=colors, width=0.55)
    ax.bar_label(
        bars,
        labels=[f"{value} ({value / counts.sum():.0%})" for value in counts.to_numpy()],
        padding=3,
        color=TEXT_PRIMARY,
    )
    ax.set_title("Situação dos alunos na base")
    ax.set_ylabel("Alunos")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=VALIDATION_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=TRAIN_COLOR),
    ]
    ax.legend(handles, ["Evasão (classe 1)", "Não evasão (classe 0)"], loc="upper right")
    return _save(fig, path)


def grade_repair(before: pd.Series, after: pd.Series, path: Path) -> Path:
    """Histograms of one grade column before and after the decimal repair."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    positive = before[before > 0]
    axes[0].hist(np.log10(positive), bins=40, color=NEUTRAL_COLOR)
    axes[0].set_title("Como veio: ordem de grandeza da nota")
    axes[0].set_xlabel("log10(nota)")
    axes[0].set_ylabel("Alunos")
    axes[1].hist(after[after > 0], bins=40, color=TRAIN_COLOR)
    axes[1].set_title("Depois do reparo: escala de 0 a 20")
    axes[1].set_xlabel("Nota média do 1º semestre (alunos com nota)")
    return _save(fig, path)


def train_vs_validation(results: pd.DataFrame, metric: str, metric_label: str, path: Path) -> Path:
    """Grouped bars comparing train and validation scores for every candidate."""
    positions = np.arange(len(results))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(results) + 1.3))
    train = ax.barh(
        positions - height / 2,
        results[f"train_{metric}"],
        height=height,
        color=TRAIN_COLOR,
        label="Treino (média dos folds)",
    )
    validation = ax.barh(
        positions + height / 2,
        results[f"validation_{metric}"],
        height=height,
        color=VALIDATION_COLOR,
        xerr=results[f"validation_{metric}_std"],
        error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1},
        label="Validação (média dos folds)",
    )
    ax.bar_label(train, fmt="%.3f", padding=4, color=TEXT_PRIMARY)
    ax.bar_label(validation, fmt="%.3f", padding=4, color=TEXT_PRIMARY)
    ax.set_yticks(positions, results["label"])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.08)
    ax.set_xlabel(metric_label)
    ax.set_title(f"{metric_label}: treino x validação cruzada", pad=28)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    return _save(fig, path)


def _score_band(ax, x, scores, color, label):
    mean = scores.mean(axis=1)
    std = scores.std(axis=1)
    ax.plot(x, mean, color=color, marker="o", label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    return mean


def validation_curve_figure(
    depths, train_scores, validation_scores, metric_label: str, path: Path
) -> Path:
    """Train and validation score of a decision tree as its depth grows."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _score_band(ax, depths, train_scores, TRAIN_COLOR, "Treino")
    validation_mean = _score_band(ax, depths, validation_scores, VALIDATION_COLOR, "Validação")
    best = int(np.argmax(validation_mean))
    ax.annotate(
        f"melhor validação\n(profundidade {depths[best]})",
        xy=(depths[best], validation_mean[best]),
        xytext=(depths[best] + 0.6, 0.79),
        color=TEXT_SECONDARY,
        fontsize=9,
        va="center",
        arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "linewidth": 1},
    )
    notes = [
        ("underfitting:\ntreino e validação baixos", (depths[0], 0.945), "left"),
        ("overfitting:\ntreino sobe, validação cai", (depths[-1], 0.885), "right"),
    ]
    for text, position, align in notes:
        ax.text(*position, text, color=TEXT_SECONDARY, fontsize=9, ha=align, va="center")
    ax.set_xticks(depths)
    ax.set_xlabel("Profundidade máxima da árvore")
    ax.set_ylabel(metric_label)
    ax.set_title("Curva de validação: do underfitting ao overfitting")
    ax.legend(loc="lower left")
    return _save(fig, path)


def learning_curve_figure(
    sizes, train_scores, validation_scores, metric_label: str, model_label: str, path: Path
) -> Path:
    """Train and validation score of the selected model as the train set grows."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _score_band(ax, sizes, train_scores, TRAIN_COLOR, "Treino")
    _score_band(ax, sizes, validation_scores, VALIDATION_COLOR, "Validação")
    ax.set_xlabel("Alunos usados no treino")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Curva de aprendizado: {model_label}")
    ax.legend(loc="lower right")
    return _save(fig, path)


def confusion_matrix_figure(matrix: np.ndarray, path: Path) -> Path:
    """Confusion matrix of the selected model on the test set."""
    labels = ["Não evasão", "Evasão"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matrix, cmap=SEQUENTIAL)
    ax.grid(visible=False)
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value}\n{value / matrix[row].sum():.0%} da linha",
                ha="center",
                va="center",
                color="#ffffff" if value > threshold else TEXT_PRIMARY,
                fontsize=11,
            )
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão no teste")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, path)


def roc_curve_figure(false_positive_rate, true_positive_rate, auc: float, path: Path) -> Path:
    """ROC curve of the selected model on the test set."""
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.plot(
        false_positive_rate, true_positive_rate, color=TRAIN_COLOR, label=f"Modelo (AUC {auc:.3f})"
    )
    ax.plot(
        [0, 1], [0, 1], color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label="Acaso (AUC 0,5)"
    )
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.set_title("Curva ROC no teste")
    ax.legend(loc="lower right")
    return _save(fig, path)


def feature_importance_figure(importances: pd.DataFrame, path: Path, top: int = 12) -> Path:
    """Permutation importance of the raw input columns on the test set."""
    ranked = importances.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(ranked) + 1.4))
    ax.barh(
        ranked["feature"],
        ranked["importance"],
        xerr=ranked["std"],
        color=TRAIN_COLOR,
        height=0.6,
        error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1},
    )
    ax.set_xlabel("Queda média no ROC AUC ao embaralhar a coluna")
    ax.set_title("Colunas que mais pesam na previsão")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
