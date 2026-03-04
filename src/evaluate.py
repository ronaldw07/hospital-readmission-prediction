"""
Model evaluation framework for hospital readmission prediction.

Computes AUC-ROC, accuracy, precision, recall, and F1 for each trained
model, generates ROC curve comparisons, and writes metrics to JSON.

Author: Ronald Wen
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_test_split(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test set.

    Author: Ronald Wen
    """
    test = pd.read_csv(data_dir / 'test.csv')
    target = 'readmitted_30'
    return test.drop(columns=[target]), test[target]


def load_models(models_dir: Path) -> dict[str, object]:
    """Load all serialised estimators from the models directory.

    Author: Ronald Wen
    """
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
    }

    models = {}
    for display_name, filename in model_files.items():
        path = models_dir / filename
        if path.exists():
            models[display_name] = joblib.load(path)
            print(f"Loaded: {display_name}")
        else:
            print(f"Warning: {path} not found — skipping.")

    return models


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_test: pd.Series, y_prob: np.ndarray) -> float:
    """Find the classification threshold that maximises F1 score.

    Sweeps the precision-recall curve to identify the operating point
    with the best harmonic mean of precision and recall.

    Args:
        y_test: True binary labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        Optimal threshold value.

    Author: Ronald Wen
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = np.where(
        (precisions + recalls) == 0,
        0,
        2 * (precisions * recalls) / (precisions + recalls),
    )
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    print(f"  Optimal threshold: {best_threshold:.3f}  (F1={f1_scores[best_idx]:.3f})")
    return best_threshold


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics at both default and optimal thresholds.

    The default 0.5 threshold is reported alongside an optimised threshold
    derived from the precision-recall curve, which better reflects
    real-world performance on imbalanced clinical data.

    Args:
        model: Fitted estimator with predict / predict_proba methods.
        X_test: Test feature matrix.
        y_test: True binary labels.

    Returns:
        Dict of metric name → value.

    Author: Ronald Wen
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    optimal_threshold = find_optimal_threshold(y_test, y_prob)
    y_pred_tuned = (y_prob >= optimal_threshold).astype(int)

    metrics = {
        'auc_roc': round(float(roc_auc_score(y_test, y_prob)), 4),
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'optimal_threshold': round(optimal_threshold, 4),
        'tuned_precision': round(float(precision_score(y_test, y_pred_tuned, zero_division=0)), 4),
        'tuned_recall': round(float(recall_score(y_test, y_pred_tuned, zero_division=0)), 4),
        'tuned_f1': round(float(f1_score(y_test, y_pred_tuned, zero_division=0)), 4),
    }
    return metrics


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Plot ROC curves for all models on a single axes for easy comparison.

    Author: Ronald Wen
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colours = ['#2196F3', '#4CAF50', '#FF5722']
    for (name, model), colour in zip(models.items(), colours):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        area = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {area:.3f})', color=colour, lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison — 30-Day Readmission', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ROC curves saved to {output_path}")


def plot_precision_recall_curves(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Plot precision-recall curves for all models on a single axes.

    PR curves are more informative than ROC curves for imbalanced datasets
    because they focus on the minority (positive) class performance.

    Author: Ronald Wen
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colours = ['#2196F3', '#4CAF50', '#FF5722']
    for (name, model), colour in zip(models.items(), colours):
        y_prob = model.predict_proba(X_test)[:, 1]
        precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recalls, precisions)
        ax.plot(recalls, precisions, label=f'{name} (AUC = {pr_auc:.3f})', color=colour, lw=2)

    baseline = y_test.mean()
    ax.axhline(baseline, color='gray', linestyle='--', lw=1, label=f'Random classifier ({baseline:.2f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve Comparison — 30-Day Readmission', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Precision-recall curves saved to {output_path}")


def plot_confusion_matrix(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot and save a confusion matrix for a single model.

    Author: Ronald Wen
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Readmitted', 'Readmitted'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate all models and persist metrics + plots.

    Author: Ronald Wen
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = data_dir or project_root / 'data' / 'processed'
    models_dir = models_dir or project_root / 'models'
    results_dir = results_dir or project_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_split(data_dir)
    models = load_models(models_dir)

    all_metrics: dict[str, dict[str, float]] = {}
    print("\n--- Model Performance ---")
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        all_metrics[name] = metrics
        print(
            f"{name:25s}  AUC={metrics['auc_roc']:.3f}  "
            f"F1={metrics['f1']:.3f} → tuned F1={metrics['tuned_f1']:.3f}  "
            f"Recall={metrics['recall']:.3f} → tuned Recall={metrics['tuned_recall']:.3f}  "
            f"Threshold={metrics['optimal_threshold']:.3f}"
        )

    # Save metrics JSON
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # ROC curves
    plot_roc_curves(models, X_test, y_test, results_dir / 'roc_curves.png')

    # Precision-recall curves
    plot_precision_recall_curves(models, X_test, y_test, results_dir / 'precision_recall_curves.png')

    # Confusion matrices per model
    for name, model in models.items():
        safe_name = name.lower().replace(' ', '_')
        plot_confusion_matrix(
            model, X_test, y_test, name,
            results_dir / f'confusion_matrix_{safe_name}.png'
        )

    return all_metrics


if __name__ == '__main__':
    run_evaluation()
