# Student Attrition & Retention AI System

**Detects student disengagement 6–8 weeks before formal academic warnings using behavioral signals — enabling proactive advisor intervention.**

> Portfolio POC · Python · scikit-learn · XGBoost · SHAP · Streamlit

---

## Problem

High dropout rates cause catastrophic tuition revenue loss. Traditional warning systems are reactive — failing grades surface weeks after disengagement began. Recovery probability drops sharply by then.

**Solution:** Behavioral signal-driven early warning that detects subtle engagement drift before any academic flag appears, surfaced through a risk-scored advisor dashboard.

## Architecture

```
OULAD CSVs (32K student records)
    → Feature Engineering (delta features, rolling windows, week-6 snapshot)
    → ML Model (XGBoost + SHAP explainability + bias audit)
    → predictions.csv (risk score 0–100, top-3 SHAP features per student)
    → Streamlit Dashboard (4 pages: overview, at-risk list, student detail, model performance)
```

## Data Sources

| Dataset | License | Size | Role |
|---------|---------|------|------|
| [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) | CC BY 4.0 | ~32K student records, 7 files | Primary — time-series behavioral logs |
| Synthetic supplement | Generated locally | 32,593 rows | financial_hold_flag, credit_load (missing from OULAD) |

**OULAD download:** Manual — requires visiting https://analyse.kmi.open.ac.uk/open_dataset and accepting the license. Place all CSV files in `data/raw/`.

**Synthetic data:** Generated automatically by `notebooks/01b_data_synthesis.ipynb` (seeded at `random_state=42`).

### File Sizes

| File | Rows | Size |
|------|------|------|
| studentVle.csv | 10,655,280 | ~400 MB raw |
| studentAssessment.csv | 173,912 | ~10 MB |
| studentInfo.csv | 32,593 | ~3 MB |
| studentRegistration.csv | 32,593 | ~2 MB |
| assessments.csv | 206 | <1 MB |
| courses.csv | 22 | <1 MB |
| vle.csv | 6,364 | <1 MB |

> **Note:** `studentVle.csv` loads to ~183 MB with dtype optimization. The pipeline aggregates it to ~2M rows immediately after load.

## Setup

```bash
git clone <repo-url>
cd student-retention-system
pip install -r requirements.txt

# 1. Download OULAD CSVs manually from https://analyse.kmi.open.ac.uk/open_dataset
#    Accept the CC BY 4.0 license, place all 7 CSVs in data/raw/

# 2. Run notebooks in order (to regenerate artifacts from scratch):
jupyter notebook
#   01a_eda.ipynb
#   01b_data_synthesis.ipynb
#   02_feature_engineering.ipynb
#   03_modeling.ipynb
#   04_explainability.ipynb
#   05_bias_audit.ipynb

# 3. Launch dashboard (pre-built artifacts already in models/ and data/output/)
streamlit run streamlit_app.py
```

## Project Structure

```
notebooks/          # Jupyter notebooks (01a_eda → 05_bias_audit)
src/                # Reusable Python modules (features.py, model.py)
data/
  raw/              # OULAD CSVs (not tracked in git)
  synthetic/        # Generated behavioral features
  processed/        # Feature matrix, labels (Phase 2 output)
  output/           # Model predictions (Phase 3 output)
docs/
  data_dictionary.md
  model_card.md
  figures/          # Plots from notebooks
models/             # Serialized trained model
streamlit_app.py    # Dashboard (Phase 4)
requirements.txt
```

## Model Results

Calibrated XGBoost, evaluated on 15% held-out test set (id_student-stratified):

| Metric | Value |
|--------|-------|
| F1 (minority / withdrawn class) | **0.6575** |
| ROC-AUC | **0.8444** |
| Precision | 0.782 |
| Recall | 0.567 |
| FPR | 0.071 |

Top predictive features (mean |SHAP|): `days_since_last_vle`, `submission_count_w6`, `studied_credits`, `avg_score_w6`, `early_submission_days`.

See [`docs/model_card.md`](docs/model_card.md) for full details including bias audit results.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Acquisition & EDA | ✅ Complete |
| 2 | Feature Engineering | ✅ Complete |
| 3 | Model Development & Validation | ✅ Complete |
| 4 | Advisor Dashboard | ✅ Complete |
| 5 | Portfolio Packaging | ✅ Complete |
