# Data Dictionary

**Date semantics (applies to all OULAD files):** `date` columns are integers representing *days relative to course start*. Negative values = days before course start. Week N = day 7×N. Training snapshot = day 42 (week 6).

---

## studentInfo.csv

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `code_module` | category | Course module identifier | 7 values: AAA, BBB, CCC, DDD, EEE, FFF, GGG |
| `code_presentation` | category | Year/semester of course delivery | 4 values: 2013B, 2013J, 2014B, 2014J. J = Oct start, B = Feb start |
| `id_student` | int32 | Student identifier | Not globally unique — same student can enroll in multiple modules |
| `gender` | category | Student gender | M / F |
| `region` | category | UK region of student | 13 regions |
| `highest_education` | category | Highest prior qualification | No Formal quals → Post Graduate Qualification (ordinal) |
| `imd_band` | category | Index of Multiple Deprivation decile | 0-10% = most deprived. **1,111 nulls (3.4%)** — add `imd_band_missing` indicator, impute mode per module-presentation group |
| `age_band` | category | Age group | 0-35, 35-55, 55<= |
| `num_of_prev_attempts` | int8 | Prior attempts at this module | 0–6 |
| `studied_credits` | int16 | OU credit weight of this enrollment | 10–360; OU-specific scale, NOT US credit hours |
| `disability` | category | Disability declaration | Y / N |
| `final_result` | category | Course outcome (TARGET) | Pass (37.9%), Withdrawn (31.2%), Fail (21.6%), Distinction (9.3%) |
| `label` | int8 | Binary target (derived) | Withdrawn=1, all others=0. Created in notebook, not in raw CSV |

---

## studentVle.csv

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `code_module` | category | Course module identifier | Foreign key to courses.csv |
| `code_presentation` | category | Course presentation | Foreign key to courses.csv |
| `id_student` | int32 | Student identifier | Foreign key to studentInfo.csv |
| `id_site` | int32 | VLE resource identifier | Foreign key to vle.csv |
| `date` | int16 | Day of activity relative to course start | Range: -25 to 269. Pre-course = negative |
| `sum_click` | int32 | Total clicks on this resource on this day | 1–6,977 per record |

**Size:** 10,655,280 rows. Load with dtype optimization (~183 MB) then immediately aggregate to daily totals per `(code_module, code_presentation, id_student, date)` — reduces to ~2M rows.

---

## studentAssessment.csv

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `id_assessment` | int32 | Assessment identifier | Foreign key to assessments.csv |
| `id_student` | int32 | Student identifier | Foreign key to studentInfo.csv |
| `date_submitted` | float32 | Day of submission relative to course start | Can be negative (early); null for non-submissions |
| `is_banked` | int8 | Reused result from prior attempt | 0/1. Banked submissions transfer score from previous enrollment |
| `score` | float32 | Assessment score | 0–100. **173 nulls (0.1%)** — non-banked nulls = non-submissions → impute 0 |

---

## assessments.csv

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `code_module` | category | Course module | |
| `code_presentation` | category | Course presentation | |
| `id_assessment` | int32 | Assessment identifier | Primary key |
| `assessment_type` | object | Type of assessment | TMA (tutor-marked, 106), CMA (computer-marked, 76), Exam (24) |
| `date` | float64 | Due date relative to course start | **11 nulls** — all Exam type (unscheduled finals); excluded by day-42 filter |
| `weight` | float64 | Contribution to final grade (%) | |

---

## courses.csv

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `code_module` | category | Course module | |
| `code_presentation` | category | Course presentation | |
| `module_presentation_length` | int64 | Course duration in days | Range: 234–269. Day 42 (week-6 cutoff) is within every course |

---

## studentRegistration.csv

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `code_module` | category | Course module | |
| `code_presentation` | category | Course presentation | |
| `id_student` | int32 | Student identifier | |
| `date_registration` | float32 | Registration day relative to course start | 45 nulls (0.1%) |
| `date_unregistration` | float32 | Withdrawal day relative to course start | **22,521 nulls (69%)** — null = student did not withdraw |

**Usage:** `date_unregistration` is NOT the label source. Used only to validate feature leakage: all features must be computed from data where `date <= 42`.

---

## vle.csv

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `id_site` | int64 | VLE resource identifier | Primary key |
| `code_module` | category | Course module | |
| `code_presentation` | category | Course presentation | |
| `activity_type` | object | Resource type | resource, oucontent, quiz, exam, forumng, homepage, etc. |
| `week_from` | float64 | Week resource becomes available | 82% null — resource available for full course duration |
| `week_to` | float64 | Week resource becomes unavailable | 82% null |

---

## data/synthetic/synthetic_features.csv

Generated by `notebooks/01b_data_synthesis.ipynb` with `numpy.random.default_rng(seed=42)`.

| Column | Type | Description | Generation logic |
|--------|------|-------------|-----------------|
| `code_module` | category | Course module | Key from studentInfo.csv |
| `code_presentation` | category | Course presentation | Key from studentInfo.csv |
| `id_student` | int32 | Student identifier | Key from studentInfo.csv |
| `financial_hold_flag` | int8 | Binary: 1 if active financial aid hold | Base prevalence 12%, OR=2.1 for withdrawal. Solved per-group probabilities: P(hold\|retained)≈0.0893, P(hold\|withdrawn)≈0.1876 |
| `credit_load` | int8 | Enrolled credit hours (US semester hours) | Discrete distribution over 9–21 credits; mode at 15. Independent of OULAD `studied_credits` |
| `credit_overload_flag` | int8 | Binary: 1 if credit_load > 18 | Derived: `(credit_load > 18).astype(int8)` |

**Join key:** `(code_module, code_presentation, id_student)` — aligns 1:1 with studentInfo.csv rows.
