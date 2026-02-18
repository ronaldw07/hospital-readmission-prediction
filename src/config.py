"""
Centralised configuration for the readmission prediction pipeline.

All hyperparameters, file paths, and runtime settings are defined here
so that changes propagate consistently across training, evaluation, and
explainability modules.

Author: Ronald Wen
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_RAW: Path = PROJECT_ROOT / 'data' / 'diabetic_data.csv'
DATA_PROCESSED: Path = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR: Path = PROJECT_ROOT / 'models'
RESULTS_DIR: Path = PROJECT_ROOT / 'results'

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

# Discharge disposition IDs that indicate death or hospice —
# readmission is not applicable for these records.
NON_READMIT_DISPOSITIONS: set[int] = {11, 13, 14, 19, 20, 21}

# Age bracket midpoints used for numeric conversion
AGE_MIDPOINTS: dict[str, int] = {
    '[0-10)': 5,
    '[10-20)': 15,
    '[20-30)': 25,
    '[30-40)': 35,
    '[40-50)': 45,
    '[50-60)': 55,
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95,
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

LOGISTIC_REGRESSION_PARAMS: dict = {
    'C': 0.1,
    'solver': 'saga',
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

RANDOM_FOREST_PARAMS: dict = {
    'n_estimators': 300,
    'max_depth': 12,
    'min_samples_leaf': 20,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

XGBOOST_PARAMS: dict = {
    'n_estimators': 400,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    # scale_pos_weight is computed dynamically from training data
}

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

SHAP_SAMPLE_SIZE: int = 2000
SHAP_MAX_DISPLAY: int = 20

SHAP_DEPENDENCE_FEATURES: list[str] = [
    'number_inpatient',
    'number_emergency',
    'time_in_hospital',
    'num_medications',
]
