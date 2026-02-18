"""
Data preprocessing pipeline for the UCI Diabetes 130-US Hospitals dataset.

This module handles all data ingestion, cleaning, feature engineering,
and train/test splitting steps before model training.

Author: Ronald Wen
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Diagnosis grouping
# ---------------------------------------------------------------------------

def _group_diagnosis(code: str) -> str:
    """Map a raw ICD-9 diagnosis code to one of nine clinical categories.

    The grouping follows standard clinical classification schemes used in
    diabetes readmission research.

    Args:
        code: Raw ICD-9 code string (may include letters, e.g. 'V30', 'E880').

    Returns:
        One of: 'Circulatory', 'Diabetes', 'Digestive', 'Genitourinary',
        'Injury', 'Musculoskeletal', 'Neoplasms', 'Respiratory', 'Other'.

    Author: Ronald Wen
    """
    if pd.isna(code) or code == '?':
        return 'Other'

    code = str(code).strip()

    # V-codes and E-codes fall into Other
    if code.startswith('V') or code.startswith('E'):
        return 'Other'

    try:
        numeric = float(code)
    except ValueError:
        return 'Other'

    if 390 <= numeric <= 459 or numeric == 785:
        return 'Circulatory'
    if (250 <= numeric < 251):
        return 'Diabetes'
    if 460 <= numeric <= 519 or numeric == 786:
        return 'Respiratory'
    if 520 <= numeric <= 579 or numeric == 787:
        return 'Digestive'
    if 580 <= numeric <= 629 or numeric == 788:
        return 'Genitourinary'
    if 140 <= numeric <= 239:
        return 'Neoplasms'
    if 710 <= numeric <= 739:
        return 'Musculoskeletal'
    if 800 <= numeric <= 999:
        return 'Injury'

    return 'Other'


# ---------------------------------------------------------------------------
# Age conversion
# ---------------------------------------------------------------------------

_AGE_MIDPOINTS: dict[str, int] = {
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


def _age_to_numeric(age_str: str) -> int:
    """Convert the age-bracket string to the bracket midpoint.

    Author: Ronald Wen
    """
    return _AGE_MIDPOINTS.get(str(age_str), 50)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load the raw CSV and perform basic sanity checks.

    Args:
        filepath: Path to diabetic_data.csv.

    Returns:
        Raw DataFrame with 101 766 rows and ~50 columns.

    Author: Ronald Wen
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download from https://archive.ics.uci.edu/dataset/296/"
            "diabetes+130-us+hospitals+for+years-1999-2008 "
            "and place it at data/diabetic_data.csv"
        )

    df = pd.read_csv(filepath, na_values=['?'])
    print(f"Loaded {len(df):,} records with {df.shape[1]} features.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data-quality filters consistent with published readmission studies.

    Steps:
      1. Remove records where discharge disposition indicates death or hospice,
         since readmission is not clinically applicable.
      2. Keep only the first encounter per patient to avoid data leakage.
      3. Drop columns with extremely high missingness or no predictive value.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.

    Author: Ronald Wen
    """
    # Discharge dispositions 11, 13, 14, 19, 20, 21 correspond to death,
    # hospice, or other non-readmissible outcomes.
    non_readmit_dispositions = {11, 13, 14, 19, 20, 21}
    df = df[~df['discharge_disposition_id'].isin(non_readmit_dispositions)].copy()

    # Keep only the first encounter per patient to prevent label leakage
    df = df.sort_values('encounter_id').drop_duplicates(subset='patient_nbr', keep='first')

    # Columns that carry no signal or are administrative proxies
    cols_to_drop = [
        'encounter_id', 'patient_nbr',
        'examide', 'citoglipton',    # near-zero variance drug columns
        'payer_code',                # high missingness, no causal link
        'weight',                    # >96% missing
        'medical_specialty',         # high cardinality + missingness
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)

    print(f"After cleaning: {len(df):,} records.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive clinically-motivated features from the raw columns.

    New features:
      - age_numeric      : numeric midpoint of age bracket
      - total_medications: sum of all drug-change columns that are non-'No'
      - total_diagnoses  : count of non-null diagnosis fields
      - diag_1_group     : ICD-9 category for primary diagnosis
      - diag_2_group     : ICD-9 category for secondary diagnosis
      - diag_3_group     : ICD-9 category for tertiary diagnosis

    Author: Ronald Wen
    """
    df = df.copy()

    # Age as a continuous proxy
    df['age_numeric'] = df['age'].apply(_age_to_numeric)

    # Count medications that were actively changed or prescribed (not 'No')
    drug_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone',
    ]
    present_drug_cols = [c for c in drug_cols if c in df.columns]
    df['total_medications'] = (
        df[present_drug_cols].apply(lambda row: (row != 'No').sum(), axis=1)
    )

    # Number of distinct diagnoses recorded at the encounter
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    present_diag_cols = [c for c in diag_cols if c in df.columns]
    df['total_diagnoses'] = df[present_diag_cols].notna().sum(axis=1)

    # Map each diagnosis field to a clinical category
    for col in present_diag_cols:
        df[f'{col}_group'] = df[col].apply(_group_diagnosis)

    # Drop the raw ICD-9 codes — too many unique values for direct encoding
    df = df.drop(columns=present_diag_cols)

    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the multi-class readmitted column to a binary label.

    Label definition:
      1 → readmitted within 30 days ('<30')
      0 → no readmission or readmission after 30 days

    This mirrors the clinical and financial threshold used by CMS for
    hospital penalty programs.

    Author: Ronald Wen
    """
    df = df.copy()
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    df = df.drop(columns=['readmitted'])
    class_counts = df['readmitted_30'].value_counts()
    readmit_rate = class_counts[1] / len(df) * 100
    print(f"30-day readmission rate: {readmit_rate:.1f}% ({class_counts[1]:,} positive cases)")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all remaining categorical columns with LabelEncoder.

    Binary-like columns (change, diabetesMed, gender) are mapped manually.
    Higher-cardinality categoricals use scikit-learn LabelEncoder so that
    the mapping stays reproducible without fitting a separate transformer.

    Author: Ronald Wen
    """
    df = df.copy()

    # Manual binary mappings
    binary_map = {'Yes': 1, 'No': 0, 'Ch': 1, 'Steady': 0,
                  'Up': 1, 'Down': 0, 'Male': 1, 'Female': 0}

    for col in ['change', 'diabetesMed']:
        if col in df.columns:
            df[col] = df[col].map({'Ch': 1, 'No': 0}).fillna(0).astype(int)

    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)

    # Label-encode remaining object columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def split_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the processed frame into train/test sets and persist them.

    Args:
        df: Fully preprocessed DataFrame (features + target column).
        output_dir: Directory where processed CSVs are written.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Author: Ronald Wen
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    target = 'readmitted_30'
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Persist for downstream use (models, evaluation, notebooks)
    pd.concat([X_train, y_train], axis=1).to_csv(output_dir / 'train.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(output_dir / 'test.csv', index=False)

    print(
        f"Train size: {len(X_train):,}  |  "
        f"Test size: {len(X_test):,}  |  "
        f"Features: {X_train.shape[1]}"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_preprocessing(
    raw_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Execute the full preprocessing pipeline end to end.

    Args:
        raw_path: Path to diabetic_data.csv. Defaults to data/diabetic_data.csv
                  relative to the project root.
        output_dir: Where to write processed splits. Defaults to data/processed/.

    Returns:
        (X_train, X_test, y_train, y_test)

    Author: Ronald Wen
    """
    project_root = Path(__file__).resolve().parent.parent
    raw_path = raw_path or project_root / 'data' / 'diabetic_data.csv'
    output_dir = output_dir or project_root / 'data' / 'processed'

    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    df = build_target(df)
    df = encode_categoricals(df)

    # Final null check — fill any remaining NaNs with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return split_and_save(df, output_dir)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = run_preprocessing()
    print("Preprocessing complete.")
