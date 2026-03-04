# Predictive Patient Readmission System

A machine learning system to predict 30-day hospital readmissions using electronic health records from 130 US hospitals.

## Overview

Hospital readmissions within 30 days represent a significant burden on healthcare systems — both financially and in terms of patient outcomes. This project builds a predictive pipeline using real-world clinical data to identify high-risk patients at discharge, enabling earlier interventions.

## Dataset

- **Source**: UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008)
- **Size**: 101,766 patient encounters across 130 hospitals
- **Features**: 50 clinical variables including demographics, diagnoses, medications, and lab results
- **Target**: Binary label — readmitted within 30 days (1) or not (0)

Download the dataset from:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years-1999-2008

Place the file at: `data/diabetic_data.csv`

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ronaldw07/hospital-readmission-prediction.git
cd hospital-readmission-prediction
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Visit the UCI repository link above and place `diabetic_data.csv` in the `data/` folder.

## Project Structure

```
hospital-readmission-prediction/
├── data/                    # Raw dataset (not tracked by git)
├── models/                  # Saved model artifacts (not tracked)
├── notebooks/
│   └── readmission_analysis.ipynb
├── results/                 # Metrics, plots, SHAP outputs
├── src/
│   ├── config.py            # Hyperparameters and paths
│   ├── utils.py             # Helper functions
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── explainability.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Running the Pipeline

```bash
# Step 1: Preprocess raw data
python src/data_preprocessing.py

# Step 2: Train all models
python src/train_models.py

# Step 3: Evaluate and compare models
python src/evaluate.py

# Step 4: Generate SHAP explainability plots
python src/explainability.py
```

Or run the full analysis interactively via the notebook:

```bash
jupyter notebook notebooks/readmission_analysis.ipynb
```

## Results

### Test Set Performance

| Model               | AUC-ROC | Recall | F1 (default) | F1 (tuned threshold) |
|---------------------|---------|--------|--------------|----------------------|
| Logistic Regression | 0.626   | 0.519  | 0.205        | 0.215                |
| Random Forest       | 0.640   | 0.453  | 0.219        | 0.221                |
| XGBoost (tuned)     | 0.650   | 0.539  | 0.223        | 0.241                |
| LightGBM            | 0.626   | 0.389  | 0.211        | 0.215                |

XGBoost was tuned via RandomizedSearchCV (30 iterations, 3-fold CV) and achieved the best AUC-ROC of 0.650. Threshold tuning via precision-recall curve analysis further improved F1 to 0.241. Training data was resampled using SMOTE to address the 9% positive class rate.

### 5-Fold Cross-Validation (Stratified)

| Model               | CV AUC-ROC     | CV F1          |
|---------------------|----------------|----------------|
| Logistic Regression | 0.622 ± 0.003  | 0.204 ± 0.001  |
| Random Forest       | 0.643 ± 0.003  | 0.220 ± 0.003  |
| XGBoost             | 0.636 ± 0.006  | 0.219 ± 0.004  |
| LightGBM            | 0.628 ± 0.004  | 0.218 ± 0.003  |

Low variance across folds confirms stable generalisation on unseen patient data.

## Key Findings

From SHAP analysis, the strongest predictors of 30-day readmission are:

1. **number_inpatient** — Prior inpatient visits are the single strongest signal
2. **number_emergency** — Emergency visit history correlates strongly with readmission risk
3. **time_in_hospital** — Longer stays indicate more complex cases
4. **num_medications** — Higher medication count reflects disease burden
5. **num_diagnoses** — Patients with more concurrent diagnoses face higher risk

## Future Improvements

- Incorporate temporal features (visit sequences per patient)
- Experiment with neural network approaches
- Build a Streamlit dashboard for clinical decision support
- Calibrate model probabilities using Platt scaling

## Author

Ronald Wen — UC Irvine, Computer Science

## License

MIT License — see [LICENSE](LICENSE) for details.
