# Model Card — Student Attrition Prediction

> Calibrated XGBoost · Binary classification (Withdrawn vs. Retained) · Week-6 snapshot

---

## Model Overview

| Attribute | Value |
|-----------|-------|
| Architecture | XGBoost classifier wrapped in imblearn Pipeline (SMOTE → XGB) |
| Calibration | Platt scaling via `CalibratedClassifierCV(method='sigmoid', cv='prefit')` |
| Output | Probability 0–1, scaled × 100 → risk score |
| Training snapshot | Day 42 (week 6 of course) |
| Artifact | `models/retention_model.pkl` |

**Model structure** (unwrap order for SHAP):
```
CalibratedClassifierCV
  └── calibrated_classifiers_[0].estimator
        └── imblearn Pipeline
              └── steps[-1][1]  →  XGBClassifier
```

---

## Training Data

| Item | Detail |
|------|--------|
| Source | OULAD (Open University Learning Analytics Dataset), CC BY 4.0 |
| Unit | `(code_module, code_presentation, id_student)` triplet |
| Total rows | 32,593 student-module pairs |
| Unique students | 28,785 (some appear in 2+ modules with different outcomes) |
| Split | 70 / 15 / 15 train/val/test, stratified by `id_student` to prevent leakage |
| Feature cutoff | `date ≤ 42` strictly applied before any aggregation |
| Class balance | 31.2% withdrawn (positive), 68.8% retained. SMOTE applied to training fold only |

---

## Features (39 total)

### VLE / Engagement (8)
| Feature | Description |
|---------|-------------|
| `total_clicks_w6` | Total VLE clicks up to day 42 |
| `active_days_w6` | Days with at least 1 VLE click, days 1–42 |
| `pre_course_clicks` | VLE clicks before course start (date < 0) |
| `vle_rolling_w6` | Sum of clicks in 7-day window ending day 42 |
| `vle_rolling_w4` | Sum of clicks in 7-day window ending day 28 |
| `days_since_last_vle` | Days elapsed since last click (at day 42) |
| `recent_silence_flag` | Binary: no VLE activity in past 14 days |
| `vle_delta_wow` | (w6 − w4) / (w4 + 1) — relative engagement trend |

### Assessment (7)
| Feature | Description |
|---------|-------------|
| `submission_count_w6` | Assignments submitted by day 42 |
| `submission_rate_w6` | Submitted / total non-exam assignments |
| `submission_rate_w4` | Same at week 4 |
| `submission_delta` | submission_rate_w6 − submission_rate_w4 |
| `avg_score_w6` | Mean score across submitted assessments |
| `gpa_trend` | OLS slope of scores across submission order |
| `early_submission_days` | Mean days before deadline (negative = late) |

### Demographic / Administrative (24)
Ordinal encodings of `imd_band`, `age_band`, `highest_education`; binary `gender_M`, `disability_Y`, `imd_band_missing`; 12 region dummies; `num_of_prev_attempts`, `studied_credits`, `days_since_registration`; synthetic `financial_hold_flag`, `credit_load`, `credit_overload_flag`.

---

## Performance Metrics

Evaluated on held-out test set (4,886 rows, 31.3% positive).

| Metric | Value |
|--------|-------|
| F1 (minority / withdrawn class) | **0.6575** |
| ROC-AUC | **0.8444** |
| Precision | 0.7823 |
| Recall | 0.5670 |
| FPR | 0.0709 |
| TP / FP / TN / FN | 859 / 239 / 3132 / 656 |

**Best hyperparameters** (RandomizedSearchCV, n_iter=20, scoring='f1'):
`n_estimators=300`, `max_depth=5`, `learning_rate=0.05`, `subsample=1.0`, `colsample_bytree=0.8`

**Comparison to baselines (validation set):**

| Model | F1 (minority) | ROC-AUC |
|-------|--------------|---------|
| Logistic Regression | ~0.58 | ~0.80 |
| Random Forest | ~0.63 | ~0.83 |
| XGBoost (calibrated) | **0.6575** | **0.8444** |

---

## Top Predictive Features (Mean |SHAP|)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | `days_since_last_vle` | 0.6683 |
| 2 | `submission_count_w6` | 0.4230 |
| 3 | `studied_credits` | 0.2930 |
| 4 | `avg_score_w6` | 0.2027 |
| 5 | `early_submission_days` | 0.1777 |
| 6 | `edu_num` | 0.1722 |
| 7 | `vle_rolling_w6` | 0.1502 |
| 8 | `active_days_w6` | 0.1501 |

Behavioral features (`days_since_last_vle`, `submission_count_w6`) dominate over demographics — the model responds primarily to in-course behavior, not student background.

---

## Known Limitations

**F1 below initial target (0.76–0.82).** Root cause: Module GGG has no assessments due before day 42 (first TMA at day 61), making `submission_count_w6 = 0` for all GGG students regardless of engagement. This degrades the assessment feature signal for ~4,500 rows. Mitigation: module-specific feature engineering or excluding GGG from assessment-based features.

**Week-6 snapshot only.** Model cannot score students before day 42. An earlier (week-4) model would require retraining with a day-28 cutoff.

**Synthetic features.** `financial_hold_flag` and `credit_load` are synthetically generated — they will not generalize to real institutional data without replacement with actual administrative records.

**OULAD is UK Open University data.** Dropout patterns, assessment structures, and demographic distributions may differ from other institutions.

**Recall = 0.567.** Model misses ~43% of students who will withdraw. At-risk identification should be used alongside, not instead of, traditional advising signals.

---

## Bias Audit Summary

Fairness evaluated across three axes. Thresholds: FPR gap ≤ 0.05, disparate impact ≤ 1.25×.

See `notebooks/05_bias_audit.ipynb` and the Model Performance page of the dashboard for group-level FPR, TPR, and prediction rate tables.

**IMD deprivation band** (primary axis): Most deprived bands (0–10%, 10–20%) show the highest withdrawal rates — the model correctly identifies higher risk in these groups. FPR gap should be verified against the 0.05 threshold in the audit notebook.

**Recommendation:** Before any deployment, run `notebooks/05_bias_audit.ipynb` on the target institution's data and apply `per_group_threshold_adjust()` if any group exceeds the FPR threshold.

---

## Intended Use

- **In scope:** Academic advisor decision support at week 6 of a course. Risk scores flag students for outreach; advisors retain full discretion over intervention decisions.
- **Out of scope:** Automated admission decisions, scholarship eligibility, or any high-stakes determination without human review.
- **Not a ground truth:** A high risk score indicates behavioral disengagement patterns consistent with historical withdrawals — it does not predict individual outcomes with certainty.

---

## Reproducibility

```bash
# Regenerate model artifact from scratch:
jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_modeling.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_explainability.ipynb
```

All randomness seeded: `numpy.random.default_rng(42)`, XGBoost `random_state=42`, split `seed=42`.
