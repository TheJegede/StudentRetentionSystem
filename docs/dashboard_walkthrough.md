# Dashboard Walkthrough

**Entry point:** `streamlit run streamlit_app.py`  
**URL:** http://localhost:8501

The dashboard surfaces week-6 attrition risk scores for 32,593 student-module pairs across 7 OULAD courses. Navigate using the sidebar radio buttons.

---

## Page 1 — Institution Overview

**Purpose:** High-level situational awareness for programme directors and institutional researchers.

**Metric strip (top row):**
- Total Enrollments — 32,593 student-module pairs
- Avg Risk Score — mean across all students
- At-Risk (≥65) — count and percentage above the default alert threshold
- High Risk (≥80) — count and percentage requiring immediate outreach
- Actual Withdrawal — ground-truth rate (31.2%) for reference

**Risk Score Distribution histogram** shows the bimodal shape: most students cluster near 0–30 (retained) with a secondary peak near 70–90 (at-risk). Dashed lines mark the 65 and 80 thresholds.

**At-Risk Rate by Module bar chart** ranks modules by percentage of students above the 65 threshold. Useful for identifying which courses need structural intervention vs. individual outreach.

**Withdrawal Rate by Demographic** (bottom row): Three side-by-side bars for IMD deprivation band, age band, and gender. IMD is the primary equity axis — lower deciles (more deprived) show higher withdrawal rates.

---

## Page 2 — At-Risk Students

**Purpose:** Prioritised student list for advisor caseload management.

**Controls:**
- **Risk Threshold slider** (50–90, default 65) — adjusts which students appear in the table in real time
- **Module dropdown** — filter to a single course module
- **Presentation dropdown** — filter to a specific year/semester cohort

**Table columns:** Student ID, Module, Presentation, Risk Score (color-coded), Predicted label, Actual outcome, Top Risk Factor, 2nd Factor, Gender, Age Band, IMD Band.

**Color coding:**
- Red background — Risk Score ≥ 80 (high risk, immediate outreach)
- Amber background — Risk Score 65–79 (at-risk, monitor)
- Green background — Risk Score < 65 (low risk)

**Navigate to Student Detail:** Select a Student ID from the dropdown at the bottom of the page and click "Go to Student Detail →". This sets the student in session state and jumps directly to the detail view.

---

## Page 3 — Student Detail

**Purpose:** Per-student drill-down for advisors preparing for a specific conversation.

**Student selector** at the top shows all students with risk ≥ 65. If a student was selected from the At-Risk page, the selector auto-focuses on that student. Students enrolled in multiple modules get an additional enrollment picker.

**Risk gauge** — Plotly indicator showing the numeric score against a green/amber/red background arc.

**Top Risk Factors (SHAP)** — Horizontal bar chart of the top 3 SHAP contributors for this specific student. Red bars push risk up; green bars push it down. Values are log-odds units (from the XGBoost tree explainer).

**Behavioral Snapshot vs Cohort Average** — Table comparing this student's key week-6 features against the cohort mean. Quickly shows where they fall below or above average.

**Behavioral Trend: Week 4 → Week 6** — Two line charts (VLE rolling clicks, submission rate) comparing this student's trajectory against the cohort average. A student dropping steeply while the cohort stays flat is the primary signal.

**Intervention Panel (mock):**
- High risk (≥80): pre-populated checkboxes for low-engagement alert, advisor appointment, financial aid review flag
- Moderate risk (65–79): engagement check-in and peer support checkboxes
- Low risk: confirmation that no action is required
- Advisor notes text area — for real deployment, wire to a CRM API; currently mock/not persisted

---

## Page 4 — Model Performance

**Purpose:** Transparency and audit page for administrators and portfolio reviewers.

**Test Set Metrics strip** — F1, ROC-AUC, Precision, Recall, FPR from the held-out test set (15%, ~4,886 rows). Expandable panel shows best XGBoost hyperparameters.

**Confusion Matrix** — Heatmap over the full dataset (32,593 rows). Rows = actual, columns = predicted.

**ROC Curve** — Full dataset curve with AUC annotation. Dashed diagonal = random baseline.

**Global Feature Importance** — Bar chart counting how often each feature appeared in a student's top-3 SHAP contributors across all 32,593 predictions. This is a proxy for global importance when the full SHAP matrix is not cached in the dashboard.

**Fairness Audit (three tabs):**
Each tab (IMD Deprivation Band, Age Band, Gender) shows:
- Max FPR Gap metric (threshold: 0.05) with PASS/FAIL indicator
- Max Disparate Impact Ratio metric (threshold: 1.25×) with PASS/FAIL indicator
- Alert banners if thresholds are exceeded
- Table of FPR, TPR, PPR, FPR_gap, DI_ratio per group
- Grouped bar chart comparing FPR and TPR across groups with the +0.05 limit line

---

## Data Dependencies

The dashboard reads three files at startup (cached after first load):

| File | Purpose |
|------|---------|
| `data/output/predictions.csv` | Risk scores, labels, top-3 SHAP per student |
| `data/raw/studentInfo.csv` | Demographics for grouping and bias audit |
| `data/processed/feature_matrix.csv` | Week-4/6 behavioral features for trend charts |
| `models/retention_model.pkl` | Test metrics and best params (metadata only — model not re-scored in dashboard) |

---

## Deployment to Streamlit Cloud

1. Push repo to GitHub (exclude `data/raw/` — too large; add to `.gitignore`)
2. Add `data/processed/`, `data/output/`, `models/` to git (these are derived artifacts, ~50 MB total)
3. Connect repo at https://share.streamlit.io
4. Set Python version 3.11, no secrets needed
5. `requirements.txt` is auto-detected

**`.gitignore` additions for deployment:**
```
data/raw/
data/synthetic/
notebooks/.ipynb_checkpoints/
src/__pycache__/
*.pyc
```
