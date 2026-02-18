"""
SHAP-based model explainability for the readmission prediction system.

Uses TreeExplainer to decompose XGBoost predictions into per-feature
Shapley value contributions, making the model's reasoning interpretable
to clinical stakeholders.

Author: Ronald Wen
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------

def load_test_split(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the preprocessed test set for SHAP analysis.

    Author: Ronald Wen
    """
    test = pd.read_csv(data_dir / 'test.csv')
    target = 'readmitted_30'
    return test.drop(columns=[target]), test[target]


def load_xgboost_model(models_dir: Path) -> object:
    """Load the serialised XGBoost model for TreeExplainer.

    Author: Ronald Wen
    """
    path = models_dir / 'xgboost.pkl'
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {path}. Run train_models.py first.")
    model = joblib.load(path)
    print(f"Loaded XGBoost model from {path}")
    return model


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: object,
    X: pd.DataFrame,
    sample_size: int = 2000,
    random_state: int = 42,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values using TreeExplainer on a random sample.

    TreeExplainer is exact for tree-based models and significantly faster
    than KernelExplainer. We subsample for interactive plotting speed —
    2 000 records is sufficient to represent the full distribution.

    Args:
        model: Fitted XGBClassifier.
        X: Feature matrix (full test set).
        sample_size: Number of rows to subsample for SHAP computation.
        random_state: Seed for subsampling reproducibility.

    Returns:
        Tuple of (shap.Explanation object, sampled feature DataFrame).

    Author: Ronald Wen
    """
    rng = np.random.default_rng(random_state)
    n = min(sample_size, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    X_sample = X.iloc[idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    print(f"SHAP values computed for {n} samples.")
    return shap_values, X_sample


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_summary(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_features: int = 20,
) -> None:
    """Generate a beeswarm summary plot of global feature importance.

    Each dot represents one patient; colour indicates feature value magnitude.
    Horizontal position shows the direction and strength of the feature's
    contribution to the readmission probability.

    Author: Ronald Wen
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_features,
        show=False,
        plot_size=None,
    )
    plt.title('SHAP Feature Importance — XGBoost Readmission Model', fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {output_path}")


def plot_bar_importance(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_features: int = 15,
) -> None:
    """Generate a bar chart of mean absolute SHAP values per feature.

    Mean |SHAP| provides a stable, model-agnostic importance ranking that
    accounts for non-linear interactions unlike standard split-based importance.

    Author: Ronald Wen
    """
    # Compute mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = X_sample.columns.tolist()

    importance_df = (
        pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
        .sort_values('mean_abs_shap', ascending=True)
        .tail(max_features)
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(
        importance_df['feature'],
        importance_df['mean_abs_shap'],
        color='#2196F3',
        edgecolor='white',
    )
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(f'Top {max_features} Features by SHAP Importance', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"SHAP bar chart saved to {output_path}")


def plot_dependence(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    feature: str,
    output_path: Path,
) -> None:
    """Create a SHAP dependence plot for a single feature.

    Dependence plots reveal how the model's reliance on a feature changes
    across its value range and whether interactions exist with a second feature
    (shown via colour).

    Author: Ronald Wen
    """
    if feature not in X_sample.columns:
        print(f"Feature '{feature}' not found — skipping dependence plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    shap.dependence_plot(
        feature,
        shap_values.values,
        X_sample,
        ax=ax,
        show=False,
    )
    ax.set_title(f'SHAP Dependence — {feature}', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"SHAP dependence plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_explainability(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    results_dir: Path | None = None,
) -> None:
    """End-to-end SHAP analysis pipeline.

    Author: Ronald Wen
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = data_dir or project_root / 'data' / 'processed'
    models_dir = models_dir or project_root / 'models'
    results_dir = results_dir or project_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_split(data_dir)
    model = load_xgboost_model(models_dir)

    shap_values, X_sample = compute_shap_values(model, X_test, sample_size=2000)

    plot_summary(shap_values, X_sample, results_dir / 'shap_summary.png')
    plot_bar_importance(shap_values, X_sample, results_dir / 'shap_importance_bar.png')

    # Dependence plots for the most clinically informative features
    for feat in ['number_inpatient', 'number_emergency', 'time_in_hospital', 'num_medications']:
        plot_dependence(shap_values, X_sample, feat, results_dir / f'shap_dependence_{feat}.png')

    print("\nExplainability analysis complete.")


if __name__ == '__main__':
    run_explainability()
