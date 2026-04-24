"""Streamlit advisor dashboard for student attrition prediction.

Pages:
  1. Institution Overview  — aggregate metrics, distributions, demographic breakdown
  2. At-Risk Students      — filterable table with threshold slider
  3. Student Detail        — risk gauge, SHAP waterfall, behavioral trend, intervention panel
  4. Model Performance     — confusion matrix, ROC, feature importance, bias audit
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(
    page_title="Student Retention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

KEY_COLS = ['code_module', 'code_presentation', 'id_student']
PAGES = ["Institution Overview", "At-Risk Students", "Student Detail", "Model Performance"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    preds    = pd.read_csv('data/output/predictions.csv')
    features = pd.read_csv('data/processed/feature_matrix.csv')
    return preds, features


@st.cache_data
def load_artifact():
    with open('models/model_metadata.json') as f:
        return json.load(f)


def fairness_metrics(df, group_col):
    rows = []
    for grp, sub in df.groupby(group_col):
        try:
            cm = confusion_matrix(sub['true_label'], sub['predicted_label'], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        except ValueError:
            continue
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppr = (tp + fp) / len(sub)
        rows.append({
            group_col: grp, 'n': int(len(sub)),
            'FPR': round(fpr, 3), 'TPR': round(tpr, 3), 'PPR': round(ppr, 3),
        })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)
    min_fpr = result['FPR'].min()
    min_ppr = result['PPR'].min() if result['PPR'].min() > 0 else 1e-9
    result['FPR_gap'] = (result['FPR'] - min_fpr).round(3)
    result['DI_ratio'] = (result['PPR'] / min_ppr).round(3)
    return result


def color_risk_cell(val):
    if not isinstance(val, (int, float)):
        return ''
    if val >= 80:
        return 'background-color: #ffb3b3; color: #cc0000; font-weight: bold'
    if val >= 65:
        return 'background-color: #fff0b3; color: #996600; font-weight: bold'
    return 'background-color: #d5f5e3; color: #1a7a3c'


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

preds, features = load_data()
master = preds  # demographics already merged into predictions.csv
artifact = load_artifact()

if 'detail_student' not in st.session_state:
    st.session_state.detail_student = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = PAGES[0]

st.sidebar.title("📊 Student Retention")
st.sidebar.caption("OULAD · 32,593 student-module pairs")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", PAGES, key='current_page')


# ===========================================================================
# PAGE 1 — Institution Overview
# ===========================================================================

if page == "Institution Overview":
    st.title("Institution Overview")
    st.caption("Week-6 snapshot · Calibrated XGBoost · 32,593 student-module pairs")

    total = len(preds)
    high_risk_n = int((preds['risk_score'] >= 80).sum())
    at_risk_n   = int((preds['risk_score'] >= 65).sum())
    avg_risk    = preds['risk_score'].mean()
    wd_rate     = preds['true_label'].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Enrollments", f"{total:,}")
    c2.metric("Avg Risk Score",    f"{avg_risk:.1f}")
    c3.metric("At-Risk  (≥65)",    f"{at_risk_n:,}",   f"{at_risk_n/total*100:.1f}%")
    c4.metric("High Risk (≥80)",   f"{high_risk_n:,}", f"{high_risk_n/total*100:.1f}%")
    c5.metric("Actual Withdrawal", f"{wd_rate:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            preds, x='risk_score', nbins=50,
            color_discrete_sequence=['#4a90d9'],
            labels={'risk_score': 'Risk Score (0–100)'},
        )
        fig.add_vline(x=65, line_dash='dash', line_color='orange',
                      annotation_text='At-Risk 65', annotation_position='top right')
        fig.add_vline(x=80, line_dash='dash', line_color='red',
                      annotation_text='High Risk 80', annotation_position='top right')
        fig.update_layout(height=340, showlegend=False, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("At-Risk Rate by Module")
        module_stats = (
            preds.groupby('code_module')['risk_score']
            .apply(lambda x: (x >= 65).mean() * 100)
            .reset_index()
            .rename(columns={'risk_score': 'At-Risk %'})
            .sort_values('At-Risk %', ascending=False)
        )
        fig2 = px.bar(
            module_stats, x='code_module', y='At-Risk %',
            color='At-Risk %', color_continuous_scale='RdYlGn_r',
            labels={'code_module': 'Module'},
            text_auto='.1f',
        )
        fig2.update_traces(textfont_size=12)
        fig2.update_layout(height=340, margin=dict(t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Withdrawal Rate by Demographic")

    d1, d2, d3 = st.columns(3)
    for col_w, grp_col, title in [
        (d1, 'imd_band',  'IMD Deprivation Band'),
        (d2, 'age_band',  'Age Band'),
        (d3, 'gender',    'Gender'),
    ]:
        grp_data = (
            master.dropna(subset=[grp_col])
            .groupby(grp_col)['true_label']
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={'true_label': 'Withdrawal Rate %'})
        )
        fig_d = px.bar(
            grp_data, x=grp_col, y='Withdrawal Rate %',
            color='Withdrawal Rate %', color_continuous_scale='RdYlGn_r',
            labels={grp_col: title}, text_auto='.1f',
        )
        fig_d.update_traces(textfont_size=11)
        fig_d.update_layout(height=280, margin=dict(t=10, b=10),
                            coloraxis_showscale=False)
        col_w.subheader(title)
        col_w.plotly_chart(fig_d, use_container_width=True)


# ===========================================================================
# PAGE 2 — At-Risk Students
# ===========================================================================

elif page == "At-Risk Students":
    st.title("At-Risk Student List")

    fc1, fc2, fc3 = st.columns([2, 2, 3])
    with fc1:
        threshold = st.slider("Risk Threshold", min_value=50, max_value=90,
                               value=65, step=5)
    with fc2:
        modules = ['All'] + sorted(master['code_module'].unique().tolist())
        sel_module = st.selectbox("Module", modules)
    with fc3:
        presentations = ['All'] + sorted(master['code_presentation'].unique().tolist())
        sel_pres = st.selectbox("Presentation", presentations)

    filtered = master[master['risk_score'] >= threshold].copy()
    if sel_module != 'All':
        filtered = filtered[filtered['code_module'] == sel_module]
    if sel_pres != 'All':
        filtered = filtered[filtered['code_presentation'] == sel_pres]
    filtered = filtered.sort_values('risk_score', ascending=False)

    st.caption(f"{len(filtered):,} students above threshold {threshold}")

    display_cols = [c for c in [
        'id_student', 'code_module', 'code_presentation',
        'risk_score', 'predicted_label', 'true_label',
        'top_shap_feature_1', 'top_shap_feature_2',
        'gender', 'age_band', 'imd_band',
    ] if c in filtered.columns]

    rename_map = {
        'id_student': 'Student ID', 'code_module': 'Module',
        'code_presentation': 'Presentation', 'risk_score': 'Risk Score',
        'predicted_label': 'Predicted', 'true_label': 'Actual',
        'top_shap_feature_1': 'Top Risk Factor',
        'top_shap_feature_2': '2nd Factor',
    }
    display_df = filtered[display_cols].rename(columns=rename_map)

    styled = display_df.style.map(color_risk_cell, subset=['Risk Score'])
    st.dataframe(styled, use_container_width=True, height=480)

    st.markdown("---")
    st.subheader("Open Student Detail")
    st.caption("Select a student then navigate to Student Detail in the sidebar, or click the button below.")

    id_options = filtered['id_student'].astype(str).tolist()
    if id_options:
        sel_id = st.selectbox("Student ID", id_options)
        if st.button("Go to Student Detail →", type="primary"):
            st.session_state.detail_student = int(sel_id)
            st.session_state.current_page = "Student Detail"
            st.rerun()
    else:
        st.info("No students above threshold with current filters.")


# ===========================================================================
# PAGE 3 — Student Detail
# ===========================================================================

elif page == "Student Detail":
    st.title("Student Detail")

    at_risk_ids = master[master['risk_score'] >= 65]['id_student'].unique()

    default_idx = 0
    if st.session_state.detail_student is not None and st.session_state.detail_student in at_risk_ids:
        default_idx = int(np.where(at_risk_ids == st.session_state.detail_student)[0][0])

    selected_id = st.selectbox("Student ID (at-risk only)", at_risk_ids, index=default_idx)
    st.session_state.detail_student = selected_id

    student_rows = master[master['id_student'] == selected_id].copy()
    feat_rows    = features[features['id_student'] == selected_id].copy()

    if len(student_rows) > 1:
        labels = [f"{r['code_module']} / {r['code_presentation']}"
                  for _, r in student_rows.iterrows()]
        sel_enr = st.selectbox("Enrollment", labels)
        idx = labels.index(sel_enr)
        row      = student_rows.iloc[idx]
        feat_row = feat_rows.iloc[idx] if idx < len(feat_rows) else feat_rows.iloc[0]
    else:
        row      = student_rows.iloc[0]
        feat_row = feat_rows.iloc[0] if len(feat_rows) > 0 else None

    risk_score = float(row['risk_score'])

    # --- Top section ---
    left, right = st.columns([1, 2])

    with left:
        bar_color = '#e74c3c' if risk_score >= 80 else '#f39c12' if risk_score >= 65 else '#27ae60'
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 16}},
            number={'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': bar_color, 'thickness': 0.25},
                'steps': [
                    {'range': [0, 65],   'color': '#d5f5e3'},
                    {'range': [65, 80],  'color': '#fef9e7'},
                    {'range': [80, 100], 'color': '#fadbd8'},
                ],
                'threshold': {
                    'line': {'color': '#555', 'width': 3},
                    'thickness': 0.75,
                    'value': 65,
                },
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=5, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        for label, key in [
            ('Module',       'code_module'),
            ('Presentation', 'code_presentation'),
            ('Gender',       'gender'),
            ('Age Band',     'age_band'),
            ('IMD Band',     'imd_band'),
            ('Predicted',    None),
            ('Actual',       None),
        ]:
            if label == 'Predicted':
                val = 'Withdrawn' if row['predicted_label'] == 1 else 'Retained'
            elif label == 'Actual':
                val = 'Withdrawn' if row['true_label'] == 1 else 'Retained'
            else:
                val = row.get(key, 'N/A')
            st.markdown(f"**{label}:** {val}")

    with right:
        st.subheader("Top Risk Factors (SHAP)")
        shap_feats = [row['top_shap_feature_1'], row['top_shap_feature_2'], row['top_shap_feature_3']]
        shap_vals  = [float(row['top_shap_value_1']), float(row['top_shap_value_2']), float(row['top_shap_value_3'])]
        colors = ['#e74c3c' if v > 0 else '#27ae60' for v in shap_vals]

        fig_shap = go.Figure(go.Bar(
            x=shap_vals[::-1],
            y=shap_feats[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"{v:+.3f}" for v in shap_vals[::-1]],
            textposition='outside',
        ))
        fig_shap.update_layout(
            height=200,
            margin=dict(t=10, b=10, l=10, r=60),
            xaxis_title='SHAP value  (red = ↑ risk,  green = ↓ risk)',
            xaxis=dict(zeroline=True, zerolinecolor='#555', zerolinewidth=1.5),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        if feat_row is not None:
            st.subheader("Behavioral Snapshot vs Cohort Average")
            key_feats = [f for f in [
                'days_since_last_vle', 'vle_rolling_w6', 'submission_rate_w6',
                'avg_score_w6', 'active_days_w6', 'submission_count_w6',
            ] if f in feat_row.index]
            cohort_mean = features[key_feats].mean()
            snap = pd.DataFrame({
                'Feature':     key_feats,
                'This Student': [round(float(feat_row[f]), 2) for f in key_feats],
                'Cohort Avg':  [round(float(cohort_mean[f]), 2) for f in key_feats],
            }).set_index('Feature')
            st.dataframe(snap, use_container_width=True)

    # --- Behavioral trend ---
    if feat_row is not None and 'vle_rolling_w4' in feat_row.index:
        st.markdown("---")
        st.subheader("Behavioral Trend: Week 4 → Week 6")

        t1, t2 = st.columns(2)
        with t1:
            vle_s = [float(feat_row['vle_rolling_w4']), float(feat_row['vle_rolling_w6'])]
            vle_c = [float(features['vle_rolling_w4'].mean()), float(features['vle_rolling_w6'].mean())]
            fig_vle = go.Figure()
            fig_vle.add_trace(go.Scatter(
                x=['Week 4', 'Week 6'], y=vle_s, mode='lines+markers',
                name='This Student', line=dict(color='#e74c3c', width=3),
                marker=dict(size=9),
            ))
            fig_vle.add_trace(go.Scatter(
                x=['Week 4', 'Week 6'], y=vle_c, mode='lines+markers',
                name='Cohort Avg', line=dict(color='#7f8c8d', dash='dash', width=2),
                marker=dict(size=7),
            ))
            fig_vle.update_layout(
                title='VLE Activity (7-day rolling clicks)',
                height=290, yaxis_title='Clicks',
                margin=dict(t=40, b=10), legend=dict(x=0.6, y=0.9),
            )
            st.plotly_chart(fig_vle, use_container_width=True)

        with t2:
            sub_s = [float(feat_row['submission_rate_w4']), float(feat_row['submission_rate_w6'])]
            sub_c = [float(features['submission_rate_w4'].mean()), float(features['submission_rate_w6'].mean())]
            fig_sub = go.Figure()
            fig_sub.add_trace(go.Scatter(
                x=['Week 4', 'Week 6'], y=sub_s, mode='lines+markers',
                name='This Student', line=dict(color='#e74c3c', width=3),
                marker=dict(size=9),
            ))
            fig_sub.add_trace(go.Scatter(
                x=['Week 4', 'Week 6'], y=sub_c, mode='lines+markers',
                name='Cohort Avg', line=dict(color='#7f8c8d', dash='dash', width=2),
                marker=dict(size=7),
            ))
            fig_sub.update_layout(
                title='Submission Rate',
                height=290, yaxis_title='Rate (0–1)',
                margin=dict(t=40, b=10), legend=dict(x=0.6, y=0.9),
            )
            st.plotly_chart(fig_sub, use_container_width=True)

    # --- Intervention panel ---
    st.markdown("---")
    st.subheader("Intervention Panel")
    i1, i2 = st.columns(2)

    with i1:
        st.markdown("**Recommended Actions**")
        if risk_score >= 80:
            st.error("High Risk — immediate outreach recommended")
            st.checkbox("Send low-engagement alert email")
            st.checkbox("Schedule advisor appointment")
            st.checkbox("Flag for financial aid review")
        elif risk_score >= 65:
            st.warning("Moderate Risk — monitor and soft-touch contact")
            st.checkbox("Send engagement check-in email")
            st.checkbox("Enroll in peer support programme")
        else:
            st.success("Low Risk — no immediate action required")

    with i2:
        st.markdown("**Advisor Notes** *(mock — not persisted)*")
        st.text_area("Notes", placeholder="Record intervention details...",
                     height=140, key="advisor_notes")
        st.button("Save Note (mock)", type="primary")


# ===========================================================================
# PAGE 4 — Model Performance
# ===========================================================================

elif page == "Model Performance":
    st.title("Model Performance & Fairness Audit")
    st.caption("Calibrated XGBoost · id_student-stratified 70/15/15 split")

    # --- Test metrics ---
    test_metrics = artifact.get('test_metrics', {})
    best_params  = artifact.get('best_params', {})

    st.subheader("Test Set Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (label, key) in zip(
        [m1, m2, m3, m4, m5],
        [('F1 (Minority)', 'f1_minority'), ('ROC-AUC', 'roc_auc'),
         ('Precision', 'precision'), ('Recall', 'recall'), ('FPR', 'fpr')],
    ):
        val = test_metrics.get(key, None)
        col.metric(label, f"{val:.4f}" if val is not None else "N/A")

    if best_params:
        with st.expander("Best XGBoost hyperparameters"):
            st.json(best_params)

    st.markdown("---")

    # --- Confusion matrix + ROC ---
    cm_col, roc_col = st.columns(2)

    y_true  = preds['true_label'].values
    y_pred  = preds['predicted_label'].values
    y_score = preds['risk_score'].values / 100.0

    with cm_col:
        st.subheader("Confusion Matrix (Full Dataset)")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Retained (0)', 'Withdrawn (1)'],
            y=['Retained (0)', 'Withdrawn (1)'],
            color_continuous_scale='Blues',
        )
        fig_cm.update_layout(height=350, margin=dict(t=30))
        st.plotly_chart(fig_cm, use_container_width=True)

    with roc_col:
        st.subheader("ROC Curve (Full Dataset)")
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        roc_val = auc(fpr_arr, tpr_arr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr_arr, y=tpr_arr, mode='lines',
            name=f'XGBoost  AUC = {roc_val:.3f}',
            line=dict(color='#e74c3c', width=2.5),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(color='gray', dash='dash'), name='Random', showlegend=False,
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=350, margin=dict(t=30),
            legend=dict(x=0.45, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # --- Global feature importance (proxy via top-3 SHAP frequency) ---
    st.markdown("---")
    st.subheader("Global Feature Importance — Appearances in Top-3 SHAP")
    st.caption("Frequency a feature ranked in each student's top-3 SHAP contributors (all 32,593 rows)")

    feat_freq = (
        pd.concat([preds['top_shap_feature_1'], preds['top_shap_feature_2'], preds['top_shap_feature_3']])
        .value_counts()
        .head(20)
        .reset_index()
    )
    feat_freq.columns = ['Feature', 'Count']

    fig_imp = px.bar(
        feat_freq, x='Count', y='Feature', orientation='h',
        color='Count', color_continuous_scale='Reds',
        labels={'Count': 'Appearances in top-3 SHAP'},
        text='Count',
    )
    fig_imp.update_traces(textposition='outside')
    fig_imp.update_layout(
        height=420, margin=dict(t=20, b=10),
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'},
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # --- Bias audit ---
    st.markdown("---")
    st.subheader("Fairness Audit")
    st.caption(
        "FPR gap threshold: 0.05 (5 pp) · Disparate impact threshold: 1.25× · "
        "Primary axis: IMD deprivation band"
    )

    tab_imd, tab_age, tab_gender = st.tabs(["IMD Deprivation Band", "Age Band", "Gender"])

    for tab, grp_col in [
        (tab_imd,    'imd_band'),
        (tab_age,    'age_band'),
        (tab_gender, 'gender'),
    ]:
        with tab:
            sub = master[[grp_col, 'true_label', 'predicted_label']].dropna()
            fair_df = fairness_metrics(sub, grp_col)

            if fair_df.empty:
                st.write("Insufficient data.")
                continue

            fpr_gap = fair_df['FPR_gap'].max()
            max_di  = fair_df['DI_ratio'].max()

            flag_fpr = fpr_gap > 0.05
            flag_di  = max_di > 1.25

            c_a, c_b = st.columns(2)
            c_a.metric("Max FPR Gap", f"{fpr_gap:.3f}",
                       delta="FAIL" if flag_fpr else "PASS",
                       delta_color="inverse")
            c_b.metric("Max Disparate Impact Ratio", f"{max_di:.2f}",
                       delta="FAIL" if flag_di else "PASS",
                       delta_color="inverse")

            if flag_fpr:
                st.error(f"FPR gap {fpr_gap:.3f} exceeds 0.05 — per-group threshold adjustment recommended")
            if flag_di:
                st.warning(f"Disparate impact {max_di:.2f}× exceeds 1.25× — review group prediction rates")

            st.dataframe(fair_df, use_container_width=True)

            fig_fair = go.Figure()
            fig_fair.add_trace(go.Bar(
                x=fair_df[grp_col].astype(str), y=fair_df['FPR'],
                name='FPR', marker_color='#e74c3c',
            ))
            fig_fair.add_trace(go.Bar(
                x=fair_df[grp_col].astype(str), y=fair_df['TPR'],
                name='TPR', marker_color='#3498db',
            ))
            fig_fair.add_hline(y=fair_df['FPR'].min() + 0.05,
                               line_dash='dash', line_color='orange',
                               annotation_text='+0.05 FPR limit')
            fig_fair.update_layout(
                barmode='group', height=300,
                xaxis_title=grp_col, yaxis_title='Rate',
                margin=dict(t=20, b=10),
                legend=dict(x=0.85, y=0.95),
            )
            st.plotly_chart(fig_fair, use_container_width=True)
