"""
Model training module for hospital readmission prediction.

Trains three classifiers — Logistic Regression, Random Forest, and XGBoost —
on the preprocessed diabetes dataset and saves each model to disk.

Author: Ronald Wen
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load train and test splits produced by the preprocessing pipeline.

    Author: Ronald Wen
    """
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')

    target = 'readmitted_30'
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    print(f"Loaded splits — train: {len(X_train):,}, test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def _class_weight_ratio(y: pd.Series) -> float:
    """Compute negative/positive ratio for XGBoost scale_pos_weight.

    The dataset is heavily imbalanced (~11% positive rate), so upweighting
    the minority class is critical for recall on readmitted patients.

    Author: Ronald Wen
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / n_pos


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit a logistic regression with L2 regularisation as a linear baseline.

    The 'saga' solver scales to large datasets and supports balanced
    class weighting out of the box.

    Author: Ronald Wen
    """
    model = LogisticRegression(
        C=0.1,
        solver='saga',
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
    )
    print("Training Logistic Regression...")
    model.fit(X_train, y_train)
    print("  Done.")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit a Random Forest with moderate depth to reduce overfitting on clinical data.

    Balanced class weighting and feature subsampling ensure diversity
    across trees while preserving signal from rare positive cases.

    Author: Ronald Wen
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
    )
    print("Training Random Forest (300 trees)...")
    model.fit(X_train, y_train)
    print("  Done.")
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> XGBClassifier:
    """Fit an XGBoost gradient-boosted classifier with explicit imbalance handling.

    scale_pos_weight is set to the negative/positive ratio so the loss
    function penalises missed positive predictions proportionally.

    Author: Ronald Wen
    """
    spw = _class_weight_ratio(y_train)

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1,
    )
    print(f"Training XGBoost (scale_pos_weight={spw:.2f})...")
    model.fit(X_train, y_train, verbose=False)
    print("  Done.")
    return model


def cross_validate_model(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Run stratified k-fold cross-validation and report mean ± std metrics.

    Stratified splits preserve the class imbalance ratio in each fold,
    giving a more reliable estimate of generalisation performance than
    a single train/test split.

    Args:
        model: Unfitted estimator to evaluate.
        X: Full feature matrix (train + test combined for CV).
        y: Full target series.
        model_name: Display name for logging.
        n_splits: Number of CV folds (default 5).
        random_state: Seed for fold reproducibility.

    Returns:
        Dict with mean and std for AUC-ROC and F1.

    Author: Ronald Wen
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {'roc_auc': 'roc_auc', 'f1': 'f1'}

    print(f"Running {n_splits}-fold CV for {model_name}...")
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    mean_auc = float(np.mean(results['test_roc_auc']))
    std_auc = float(np.std(results['test_roc_auc']))
    mean_f1 = float(np.mean(results['test_f1']))
    std_f1 = float(np.std(results['test_f1']))

    print(
        f"  {model_name}: AUC={mean_auc:.3f} ± {std_auc:.3f}  "
        f"F1={mean_f1:.3f} ± {std_f1:.3f}"
    )

    return {
        'cv_auc_mean': round(mean_auc, 4),
        'cv_auc_std': round(std_auc, 4),
        'cv_f1_mean': round(mean_f1, 4),
        'cv_f1_std': round(std_f1, 4),
    }


def save_model(model: object, name: str, models_dir: Path) -> Path:
    """Serialize a fitted estimator to disk using joblib.

    Author: Ronald Wen
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f'{name}.pkl'
    joblib.dump(model, out_path)
    print(f"Saved: {out_path}")
    return out_path


def run_training(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    run_cv: bool = True,
) -> dict[str, object]:
    """Train all three models, run cross-validation, and persist to disk.

    Args:
        data_dir: Directory containing processed train/test CSVs.
        models_dir: Directory to write serialised model files.
        run_cv: Whether to run 5-fold CV before final training (default True).

    Author: Ronald Wen
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = data_dir or project_root / 'data' / 'processed'
    models_dir = models_dir or project_root / 'models'

    X_train, X_test, y_train, y_test = load_splits(data_dir)

    model_builders = {
        'logistic_regression': lambda: train_logistic_regression(X_train, y_train),
        'random_forest': lambda: train_random_forest(X_train, y_train),
        'xgboost': lambda: train_xgboost(X_train, y_train),
    }

    cv_estimators = {
        'logistic_regression': LogisticRegression(C=0.1, solver='saga', max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1),
        'random_forest': RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1),
        'xgboost': XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=_class_weight_ratio(y_train), eval_metric='logloss', random_state=42, n_jobs=-1),
    }

    if run_cv:
        print("\n--- 5-Fold Cross-Validation ---")
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])
        for name, estimator in cv_estimators.items():
            cross_validate_model(estimator, X_all, y_all, model_name=name)

    print("\n--- Final Model Training ---")
    models = {name: builder() for name, builder in model_builders.items()}

    for name, model in models.items():
        save_model(model, name, models_dir)

    print("\nAll models trained and saved.")
    return models


if __name__ == '__main__':
    run_training()
