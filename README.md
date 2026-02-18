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

| Model               | AUC-ROC | Accuracy | Precision | Recall | F1    |
|---------------------|---------|----------|-----------|--------|-------|
| Logistic Regression | 0.641   | 0.889    | 0.421     | 0.078  | 0.131 |
| Random Forest       | 0.648   | 0.889    | 0.457     | 0.074  | 0.127 |
| XGBoost             | 0.672   | 0.888    | 0.432     | 0.093  | 0.153 |

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
