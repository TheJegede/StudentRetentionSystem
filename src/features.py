"""Feature engineering for student attrition prediction.

Entry point: build_feature_matrix()
Training snapshot: day 42 (week 6). All behavioral features computed from date <= 42.
Training unit: (code_module, code_presentation, id_student) triplets — NOT unique students.
"""

from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF_DAY = 42
KEY_COLS = ['code_module', 'code_presentation', 'id_student']

_VLE_DTYPES = {
    'code_module': 'category', 'code_presentation': 'category',
    'id_student': 'int32', 'id_site': 'int32',
    'date': 'int16', 'sum_click': 'int32',
}
_INFO_DTYPES = {
    'code_module': 'category', 'code_presentation': 'category',
    'id_student': 'int32', 'gender': 'category', 'region': 'category',
    'highest_education': 'category', 'imd_band': 'category', 'age_band': 'category',
    'num_of_prev_attempts': 'int8', 'studied_credits': 'int16',
    'disability': 'category', 'final_result': 'category',
}

_IMD_ORDER = {
    '0-10%': 1, '10-20%': 2, '20-30%': 3, '30-40%': 4, '40-50%': 5,
    '50-60%': 6, '60-70%': 7, '70-80%': 8, '80-90%': 9, '90-100%': 10,
}
_AGE_ORDER = {'0-35': 1, '35-55': 2, '55<=': 3}
_EDU_ORDER = {
    'No Formal quals': 0, 'Lower Than A Level': 1, 'A Level or Equivalent': 2,
    'HE Qualification': 3, 'Post Graduate Qualification': 4,
}


def _load_data(data_dir: Path, synth_dir: Path) -> dict:
    data = {}
    data['info'] = pd.read_csv(data_dir / 'studentInfo.csv', dtype=_INFO_DTYPES)

    vle_raw = pd.read_csv(data_dir / 'studentVle.csv', dtype=_VLE_DTYPES)
    data['daily_vle'] = (
        vle_raw
        .groupby(KEY_COLS + ['date'], observed=True)['sum_click']
        .sum()
        .reset_index()
    )
    del vle_raw

    data['student_assess'] = pd.read_csv(
        data_dir / 'studentAssessment.csv',
        dtype={'id_assessment': 'int32', 'id_student': 'int32',
               'date_submitted': 'float32', 'is_banked': 'int8', 'score': 'float32'},
    )
    data['assessments'] = pd.read_csv(data_dir / 'assessments.csv')
    data['registration'] = pd.read_csv(
        data_dir / 'studentRegistration.csv',
        dtype={'code_module': 'category', 'code_presentation': 'category',
               'id_student': 'int32', 'date_registration': 'float32',
               'date_unregistration': 'float32'},
    )
    data['synth'] = pd.read_csv(
        synth_dir / 'synthetic_features.csv',
        dtype={'code_module': 'category', 'code_presentation': 'category',
               'id_student': 'int32', 'financial_hold_flag': 'int8',
               'credit_load': 'int8', 'credit_overload_flag': 'int8'},
    )
    return data


def _vle_features(daily_vle: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    in_course = daily_vle[(daily_vle['date'] >= 0) & (daily_vle['date'] <= cutoff_day)]
    pre_course = daily_vle[daily_vle['date'] < 0]

    total = (
        in_course.groupby(KEY_COLS, observed=True)['sum_click']
        .agg(total_clicks_w6='sum', active_days_w6='count')
        .reset_index()
    )

    pre = (
        pre_course.groupby(KEY_COLS, observed=True)['sum_click']
        .sum().reset_index()
        .rename(columns={'sum_click': 'pre_course_clicks'})
    )

    # 7-day rolling windows: week 6 = days 36-42, week 4 = days 22-28
    w6_start, w4_start = cutoff_day - 6, cutoff_day - 20
    w6_roll = (
        daily_vle[(daily_vle['date'] >= w6_start) & (daily_vle['date'] <= cutoff_day)]
        .groupby(KEY_COLS, observed=True)['sum_click'].sum().reset_index()
        .rename(columns={'sum_click': 'vle_rolling_w6'})
    )
    w4_roll = (
        daily_vle[(daily_vle['date'] >= w4_start) & (daily_vle['date'] <= w4_start + 6)]
        .groupby(KEY_COLS, observed=True)['sum_click'].sum().reset_index()
        .rename(columns={'sum_click': 'vle_rolling_w4'})
    )

    last_active = (
        in_course.groupby(KEY_COLS, observed=True)['date']
        .max().reset_index()
        .rename(columns={'date': 'last_vle_day'})
    )

    # recent_silence: zero VLE in last 14 days before cutoff (days 29-42)
    recent_clicks = (
        daily_vle[(daily_vle['date'] >= cutoff_day - 13) & (daily_vle['date'] <= cutoff_day)]
        .groupby(KEY_COLS, observed=True)['sum_click'].sum().reset_index()
        .rename(columns={'sum_click': '_recent'})
    )

    feat = total.copy()
    for df in [pre, w6_roll, w4_roll, last_active, recent_clicks]:
        feat = feat.merge(df, on=KEY_COLS, how='left')

    feat['pre_course_clicks'] = feat['pre_course_clicks'].fillna(0).astype('int32')
    feat['vle_rolling_w6'] = feat['vle_rolling_w6'].fillna(0).astype('int32')
    feat['vle_rolling_w4'] = feat['vle_rolling_w4'].fillna(0).astype('int32')
    feat['last_vle_day'] = feat['last_vle_day'].fillna(-1)
    feat['days_since_last_vle'] = (cutoff_day - feat['last_vle_day']).clip(lower=0).astype('int16')
    feat['recent_silence_flag'] = (feat['_recent'].isna() | (feat['_recent'] == 0)).astype('int8')
    feat['active_days_w6'] = feat['active_days_w6'].fillna(0).astype('int16')
    feat['total_clicks_w6'] = feat['total_clicks_w6'].fillna(0).astype('int32')
    # wow delta: signed % change; denominator +1 avoids div-by-zero
    feat['vle_delta_wow'] = (
        (feat['vle_rolling_w6'] - feat['vle_rolling_w4']) / (feat['vle_rolling_w4'] + 1)
    ).astype('float32')

    return feat.drop(columns=['_recent', 'last_vle_day'])


def _assessment_features(
    student_assess: pd.DataFrame, assessments: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    # Total non-exam assessments per module-presentation = denominator for submission rates.
    # Most assessments are due after day 42, so we use total course assessments, not "due by 42".
    # This makes GGG (first TMA due day 61) comparable to other modules.
    n_total = (
        assessments[assessments['assessment_type'] != 'Exam']
        .groupby(['code_module', 'code_presentation'])['id_assessment']
        .count().reset_index()
        .rename(columns={'id_assessment': 'n_total_assessments'})
    )
    n_total['code_module'] = n_total['code_module'].astype('category')
    n_total['code_presentation'] = n_total['code_presentation'].astype('category')

    # Join submissions with assessment metadata
    meta = assessments[['id_assessment', 'code_module', 'code_presentation',
                         'assessment_type', 'date']].copy()
    meta = meta.rename(columns={'date': 'due_date'})
    meta['code_module'] = meta['code_module'].astype('category')
    meta['code_presentation'] = meta['code_presentation'].astype('category')

    sub = student_assess.merge(meta, on='id_assessment', how='left')
    sub['id_student'] = sub['id_student'].astype('int32')

    # Exclude exam submissions (CMA/TMA only for behavioral counts)
    sub_no_exam = sub[sub['assessment_type'] != 'Exam']

    # Submissions by week 6 cutoff and week 4 (day 28)
    sub_w6 = sub_no_exam[sub_no_exam['date_submitted'] <= cutoff_day]
    sub_w4 = sub_no_exam[sub_no_exam['date_submitted'] <= cutoff_day - 14]

    cnt_w6 = (
        sub_w6.groupby(KEY_COLS, observed=True)['id_assessment']
        .count().reset_index()
        .rename(columns={'id_assessment': 'submission_count_w6'})
    )
    cnt_w4 = (
        sub_w4.groupby(KEY_COLS, observed=True)['id_assessment']
        .count().reset_index()
        .rename(columns={'id_assessment': 'submission_count_w4'})
    )

    # Avg score: non-null scores in submissions by cutoff
    scored = sub_w6[sub_w6['score'].notna()]
    avg_score = (
        scored.groupby(KEY_COLS, observed=True)['score']
        .mean().reset_index()
        .rename(columns={'score': 'avg_score_w6'})
    )

    # GPA trend: linear slope of scores over submission dates (vectorized OLS)
    s = scored.copy()
    s['xy'] = s['date_submitted'] * s['score']
    s['x2'] = s['date_submitted'] ** 2
    slope_grp = s.groupby(KEY_COLS, observed=True).agg(
        n=('score', 'count'),
        sum_x=('date_submitted', 'sum'),
        sum_y=('score', 'sum'),
        sum_xy=('xy', 'sum'),
        sum_x2=('x2', 'sum'),
    ).reset_index()
    denom = slope_grp['n'] * slope_grp['sum_x2'] - slope_grp['sum_x'] ** 2
    slope_grp['gpa_trend'] = np.where(
        (slope_grp['n'] >= 2) & (denom > 0),
        (slope_grp['n'] * slope_grp['sum_xy'] - slope_grp['sum_x'] * slope_grp['sum_y']) / denom,
        0.0,
    ).astype('float32')
    gpa = slope_grp[KEY_COLS + ['gpa_trend']]

    # Early submission days: mean(due_date - date_submitted) for subs w/ known due date
    timely = sub_w6[sub_w6['due_date'].notna()].copy()
    timely['days_early'] = timely['due_date'] - timely['date_submitted']
    early = (
        timely.groupby(KEY_COLS, observed=True)['days_early']
        .mean().reset_index()
        .rename(columns={'days_early': 'early_submission_days'})
    )

    # Merge everything
    feat = cnt_w6.copy()
    for df in [cnt_w4, avg_score, gpa, early]:
        feat = feat.merge(df, on=KEY_COLS, how='left')

    feat['submission_count_w6'] = feat['submission_count_w6'].fillna(0).astype('int16')
    feat['submission_count_w4'] = feat['submission_count_w4'].fillna(0).astype('int16')
    feat['avg_score_w6'] = feat['avg_score_w6'].fillna(feat['avg_score_w6'].median()).astype('float32')
    feat['gpa_trend'] = feat['gpa_trend'].fillna(0.0)
    feat['early_submission_days'] = feat['early_submission_days'].fillna(0.0).astype('float32')

    # Merge n_total denominator and compute rates
    feat = feat.merge(n_total, on=['code_module', 'code_presentation'], how='left')
    denom_col = feat['n_total_assessments'].clip(lower=1)
    feat['submission_rate_w6'] = (feat['submission_count_w6'] / denom_col).clip(0, 1).astype('float32')
    feat['submission_rate_w4'] = (feat['submission_count_w4'] / denom_col).clip(0, 1).astype('float32')
    feat['submission_delta'] = (feat['submission_rate_w6'] - feat['submission_rate_w4']).astype('float32')

    return feat[KEY_COLS + [
        'submission_count_w6', 'submission_rate_w6', 'submission_rate_w4',
        'submission_delta', 'avg_score_w6', 'gpa_trend', 'early_submission_days',
    ]]


def _demographic_features(
    info: pd.DataFrame, synth: pd.DataFrame, registration: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    feat = info[KEY_COLS + [
        'gender', 'region', 'highest_education', 'imd_band',
        'age_band', 'disability', 'num_of_prev_attempts', 'studied_credits',
    ]].copy()

    feat['gender_M'] = (feat['gender'] == 'M').astype('int8')
    feat['disability_Y'] = (feat['disability'] == 'Y').astype('int8')

    feat['imd_band_missing'] = feat['imd_band'].isna().astype('int8')
    feat['imd_band_num'] = feat['imd_band'].map(_IMD_ORDER).fillna(5.0).astype('float32')
    feat['age_band_num'] = feat['age_band'].map(_AGE_ORDER).astype('float32')
    feat['edu_num'] = feat['highest_education'].map(_EDU_ORDER).astype('float32')

    region_dummies = pd.get_dummies(feat['region'], prefix='region', drop_first=True, dtype='int8')
    feat = pd.concat([feat, region_dummies], axis=1)

    reg = registration[KEY_COLS + ['date_registration']].copy()
    reg['days_since_registration'] = cutoff_day - reg['date_registration']
    feat = feat.merge(reg[KEY_COLS + ['days_since_registration']], on=KEY_COLS, how='left')
    feat['days_since_registration'] = feat['days_since_registration'].fillna(
        feat['days_since_registration'].median()
    ).astype('float32')

    feat = feat.merge(
        synth[KEY_COLS + ['financial_hold_flag', 'credit_load', 'credit_overload_flag']],
        on=KEY_COLS, how='left',
    )

    drop_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    return feat.drop(columns=drop_cols)


def build_feature_matrix(
    data_dir: str | Path = 'data/raw',
    synth_dir: str | Path = 'data/synthetic',
    cutoff_day: int = CUTOFF_DAY,
    save_dir: str | Path = 'data/processed',
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and binary labels at cutoff_day snapshot.

    Saves feature_matrix.csv and labels.csv to save_dir.
    Returns (X, y) indexed by KEY_COLS.
    """
    data_dir, synth_dir, save_dir = Path(data_dir), Path(synth_dir), Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data...')
    data = _load_data(data_dir, synth_dir)
    info = data['info']
    n_expected = len(info)

    print('Computing VLE features...')
    vle_feat = _vle_features(data['daily_vle'], cutoff_day)

    print('Computing assessment features...')
    assess_feat = _assessment_features(data['student_assess'], data['assessments'], cutoff_day)

    print('Computing demographic features...')
    demo_feat = _demographic_features(info, data['synth'], data['registration'], cutoff_day)

    print('Merging...')
    X = info[KEY_COLS].copy()
    for feat_df in [vle_feat, assess_feat, demo_feat]:
        X = X.merge(feat_df, on=KEY_COLS, how='left')

    X = X.set_index(KEY_COLS)

    # Fill NaN for students with zero VLE activity (valid — not missing data)
    zero_fill_vle = ['total_clicks_w6', 'active_days_w6', 'pre_course_clicks',
                     'vle_rolling_w6', 'vle_rolling_w4', 'vle_delta_wow']
    for col in zero_fill_vle:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    X['days_since_last_vle'] = X['days_since_last_vle'].fillna(cutoff_day).astype('int16')
    X['recent_silence_flag'] = X['recent_silence_flag'].fillna(1).astype('int8')

    # Fill NaN for students with zero submissions
    zero_fill_assess = ['submission_count_w6', 'submission_rate_w6', 'submission_rate_w4',
                         'submission_delta', 'gpa_trend', 'early_submission_days']
    for col in zero_fill_assess:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    if 'avg_score_w6' in X.columns:
        X['avg_score_w6'] = X['avg_score_w6'].fillna(X['avg_score_w6'].median())

    y = (info.set_index(KEY_COLS)['final_result'] == 'Withdrawn').astype('int8')
    y.name = 'label'

    assert len(X) == n_expected, f'Row count: {len(X)} vs {n_expected}'

    null_summary = X.isnull().sum()
    if null_summary.sum() > 0:
        print('Remaining nulls:')
        print(null_summary[null_summary > 0])

    print(f'\nFeature matrix: {X.shape[0]:,} rows × {X.shape[1]} features')
    print(f'Label: {y.sum():,} withdrawn / {(y==0).sum():,} retained '
          f'({y.mean()*100:.1f}% positive)')

    X.reset_index().to_csv(save_dir / 'feature_matrix.csv', index=False)
    y.reset_index().to_csv(save_dir / 'labels.csv', index=False)
    print(f'Saved to {save_dir}/')

    return X, y
