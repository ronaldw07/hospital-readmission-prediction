"""
Model training module for hospital readmission prediction.

Trains four classifiers — Logistic Regression, Random Forest, XGBoost,
and LightGBM — on SMOTE-resampled training data and saves each model to disk.
XGBoost is tuned via RandomizedSearchCV to maximise AUC-ROC.

Author: Ronald Wen
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
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


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to oversample the minority class in the training set.

    SMOTE (Synthetic Minority Oversampling Technique) generates synthetic
    positive examples by interpolating between existing minority samples,
    producing a balanced training set without simply duplicating records.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.
        random_state: Seed for reproducibility.

    Returns:
        Resampled (X_train_smote, y_train_smote) with balanced class distribution.

    Author: Ronald Wen
    """
    smote = SMOTE(random_state=random_state, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    print(f"After SMOTE — train size: {len(X_res):,}  (class balance: {y_res.mean():.2f})")
    return X_res, y_res


def _class_weight_ratio(y: pd.Series) -> float:
    """Compute negative/positive ratio for XGBoost/LightGBM scale_pos_weight.

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
    """Fit a Random Forest with moderate depth to reduce overfitting.

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
    n_iter: int = 30,
) -> XGBClassifier:
    """Tune and fit an XGBoost classifier via RandomizedSearchCV.

    Searches 30 random hyperparameter combinations using 3-fold stratified CV
    optimising AUC-ROC, then retrains the best configuration on the full
    training set.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.
        random_state: Seed for reproducibility.
        n_iter: Number of random hyperparameter combinations to try.

    Author: Ronald Wen
    """
    spw = _class_weight_ratio(y_train)

    param_dist = {
        'n_estimators': [200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [4, 5, 6, 7, 8],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5, 10],
        'gamma': [0, 0.1, 0.2, 0.5],
    }

    base = XGBClassifier(
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    print(f"Tuning XGBoost via RandomizedSearchCV ({n_iter} iterations)...")
    search.fit(X_train, y_train)
    print(f"  Best AUC-ROC (CV): {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LGBMClassifier:
    """Fit a LightGBM gradient-boosted classifier.

    LightGBM uses histogram-based learning and leaf-wise tree growth,
    making it faster and often more accurate than XGBoost on tabular data.
    scale_pos_weight handles the class imbalance in the loss function.

    Author: Ronald Wen
    """
    spw = _class_weight_ratio(y_train)

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    print("Training LightGBM (500 estimators)...")
    model.fit(X_train, y_train)
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
    """Train all four models on SMOTE-resampled data and persist to disk.

    Pipeline:
      1. Load train/test splits
      2. Apply SMOTE to balance the training set
      3. Run 5-fold CV on original (non-SMOTE) training data for unbiased estimates
      4. Train final models on SMOTE data
      5. Persist all models

    Args:
        data_dir: Directory containing processed train/test CSVs.
        models_dir: Directory to write serialised model files.
        run_cv: Whether to run 5-fold CV (default True).

    Author: Ronald Wen
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = data_dir or project_root / 'data' / 'processed'
    models_dir = models_dir or project_root / 'models'

    X_train, X_test, y_train, y_test = load_splits(data_dir)

    # Apply SMOTE to create balanced training data for final models
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    cv_estimators = {
        'logistic_regression': LogisticRegression(C=0.1, solver='saga', max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1),
        'random_forest': RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1),
        'xgboost': XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=_class_weight_ratio(y_train), eval_metric='logloss', random_state=42, n_jobs=-1),
        'lightgbm': LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=63, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=_class_weight_ratio(y_train), random_state=42, n_jobs=-1, verbose=-1),
    }

    if run_cv:
        print("\n--- 5-Fold Cross-Validation (original training data) ---")
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])
        for name, estimator in cv_estimators.items():
            cross_validate_model(estimator, X_all, y_all, model_name=name)

    print("\n--- Final Model Training (SMOTE-resampled data) ---")
    models = {
        'logistic_regression': train_logistic_regression(X_train_smote, y_train_smote),
        'random_forest': train_random_forest(X_train_smote, y_train_smote),
        'xgboost': train_xgboost(X_train_smote, y_train_smote),
        'lightgbm': train_lightgbm(X_train_smote, y_train_smote),
    }

    for name, model in models.items():
        save_model(model, name, models_dir)

    print("\nAll models trained and saved.")
    return models


if __name__ == '__main__':
    run_training()
