import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Burnout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
DARK = '#0e1117'
C3   = ['#22c55e', '#f59e0b', '#ef4444']
ICON = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
MSG  = {
    'Low'   : '✅ Low burnout. Great habits — keep it up!',
    'Medium': '⚡ Moderate burnout. Focus on stress management and sleep.',
    'High'  : '🚨 High burnout detected. Please seek support and reduce workload.',
}

# LabelEncoder sorts alphabetically — must match training
GENDER_MAP = {'Female': 0, 'Male': 1, 'Other': 2}
COURSE_MAP = {'BBA': 0, 'BCA': 1, 'BSc': 2, 'BTech': 3, 'MBA': 4, 'MCA': 5}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    meta  = json.load(open('model_artifacts/feature_meta.json'))
    model = joblib.load('model_artifacts/best_model.pkl')
    sc    = joblib.load('model_artifacts/scaler.pkl')
    return meta, model, sc

meta, model, sc = load()
feat_cols  = meta['feature_columns']
target_inv = meta['target_map_inv']
res        = meta['results']
best_name  = meta['best_model']
test_m     = meta['test_metrics']

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Burnout Predictor")
    st.markdown("---")
    page = st.radio(
        "",
        ["🔮 Predict", "🏆 Model Results"],
        label_visibility='collapsed'
    )
    st.markdown("---")
    st.caption(f"Best model: **{best_name}**")
    st.metric("Test Accuracy",         f"{test_m['accuracy']:.4f}")
    st.metric("Test F1 (weighted)",    f"{test_m['f1_weighted']:.4f}")
    st.metric("Test F1 (macro)",       f"{test_m['f1_macro']:.4f}")

# ── PAGE 1: PREDICT ───────────────────────────────────────────────────────────
if page == "🔮 Predict":
    st.title("🔮 Predict Your Burnout Level")
    st.markdown("Fill in your details below and click **Predict** to see your burnout risk.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("👤 Personal")
        age    = st.slider("Age", 17, 25, 21)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox("Course", ["BTech", "BCA", "BSc", "MBA", "BBA", "MCA"])
        year   = st.selectbox("Year", ["1st", "2nd", "3rd", "4th"])

    with c2:
        st.subheader("📚 Academic")
        study_h    = st.slider("Daily Study Hours",       0.0, 12.0,  5.0, 0.5)
        cgpa       = st.slider("CGPA",                    4.0, 10.0,  7.0, 0.1)
        attendance = st.slider("Attendance %",           50.0,100.0, 75.0, 0.5)
        acad_p     = st.slider("Academic Pressure (1–10)", 1,   10,    5)
        inet_q     = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])

    with c3:
        st.subheader("🧘 Wellbeing")
        sleep_h  = st.slider("Daily Sleep Hours",           3.0, 10.0, 7.0, 0.5)
        sleep_q  = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
        screen_h = st.slider("Screen Time (hours/day)",     0.0, 12.0, 4.0, 0.5)
        stress   = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        anxiety  = st.slider("Anxiety Score (1–10)",          1,   10,   5)
        depress  = st.slider("Depression Score (1–10)",       1,   10,   5)
        fin_s    = st.slider("Financial Stress (1–10)",       1,   10,   5)
        social_s = st.slider("Social Support (1–10)",         1,   10,   5)
        phys_a   = st.slider("Physical Activity (hrs/day)", 0.0,  4.0, 1.0, 0.5)

    st.markdown("---")

    if st.button("🚀 Predict My Burnout Level", use_container_width=True, type="primary"):
        om       = meta['ordinal_maps']
        stress_n = om['stress_level'][stress]
        sleep_qn = om['sleep_quality'][sleep_q]
        inet_n   = om['internet_quality'][inet_q]
        year_n   = om['year'][year]

        row = {
            'age'                     : age,
            'gender'                  : GENDER_MAP[gender],
            'course'                  : COURSE_MAP[course],
            'year'                    : year_n,
            'daily_study_hours'       : study_h,
            'daily_sleep_hours'       : sleep_h,
            'screen_time_hours'       : screen_h,
            'stress_level'            : stress_n,
            'anxiety_score'           : anxiety,
            'depression_score'        : depress,
            'academic_pressure_score' : acad_p,
            'financial_stress_score'  : fin_s,
            'social_support_score'    : social_s,
            'physical_activity_hours' : phys_a,
            'sleep_quality'           : sleep_qn,
            'attendance_percentage'   : attendance,
            'cgpa'                    : cgpa,
            'internet_quality'        : inet_n,
        }

        # Engineered features — must match training exactly
        row['mental_composite']      = anxiety + depress + stress_n * 2
        row['stress_resilience']     = social_s / (stress_n + 1)
        row['sleep_deficit']         = max(0, 8.0 - sleep_h)
        row['study_overload']        = study_h / (sleep_h + 0.5)
        row['academic_mental_cross'] = acad_p * depress
        row['recovery_index']        = phys_a * 2 + sleep_h + social_s * 0.5
        row['burnout_risk_raw']      = anxiety + depress + acad_p
        row['financial_mental']      = fin_s * anxiety
        row['screen_to_sleep']       = screen_h / (sleep_h + 0.1)
        row['cgpa_efficiency']       = cgpa / (study_h + 0.5)
        row['total_stress_burden']   = acad_p + fin_s + stress_n * 3
        row['wellbeing_gap']         = row['mental_composite'] - row['recovery_index']

        Xi       = pd.DataFrame([row])[feat_cols]
        pred_cls = model.predict(Xi)[0]
        proba    = model.predict_proba(Xi)[0]
        label    = target_inv[str(pred_cls)]

        # Result header
        st.markdown(
            f"## {ICON[label]}  {label} Burnout  —  "
            f"confidence: {max(proba) * 100:.1f}%"
        )

        # Metric row
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction",  label)
        m2.metric("Probability", f"{max(proba) * 100:.1f}%")
        m3.metric("Confidence",
                  "High" if max(proba) > 0.8 else
                  "Medium" if max(proba) > 0.6 else "Low")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=DARK)
        ax.set_facecolor(DARK)
        bars = ax.barh(['Low', 'Medium', 'High'], proba, color=C3, edgecolor=DARK)
        for b in bars:
            ax.text(
                b.get_width() + .01,
                b.get_y() + b.get_height() / 2,
                f'{b.get_width():.1%}',
                va='center', color='white', fontsize=11
            )
        ax.set_xlim(0, 1.2)
        ax.tick_params(colors='white')
        ax.set_xlabel('Probability', color='white')
        st.pyplot(fig)

        # Message
        if   label == 'High':   st.error(MSG[label])
        elif label == 'Medium': st.warning(MSG[label])
        else:                   st.success(MSG[label])

# ── PAGE 2: MODEL RESULTS ─────────────────────────────────────────────────────
elif page == "🏆 Model Results":
    st.title("🏆 Model Performance")

    st.markdown(
        f"**Best model:** {best_name} &nbsp;|&nbsp; "
        f"Test Accuracy: `{test_m['accuracy']:.4f}` &nbsp;|&nbsp; "
        f"F1 (weighted): `{test_m['f1_weighted']:.4f}`"
    )
    st.markdown("---")

    rdf = pd.DataFrame(res).T.sort_values('F1 Weighted', ascending=False)
    st.dataframe(
        rdf.style.background_gradient(
            subset=['Accuracy', 'F1 Weighted', 'F1 Macro'],
            cmap='RdYlGn'
        ),
        use_container_width=True
    )

    for img, title in [
        ('confusion_matrices.png',  'Confusion Matrices'),
        ('feature_importance.png',  'Feature Importance'),
        ('eda_new_labels.png',      'Burnout Score Distribution'),
    ]:
        if os.path.exists(img):
            st.markdown("---")
            st.subheader(title)
            st.image(img, use_column_width=True)
