import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Patch all required custom functions before loading model ──────────────────
import sys

def to_str_array(x):
    x = pd.DataFrame(x).astype(object)
    x = x.where(~x.isna(), "__MISSING__")
    return x.astype(str).values

# Patch into all possible module locations joblib might look
sys.modules["__main__"].to_str_array = to_str_array

# Also patch imblearn in case it's referenced
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maternity Risk Predictor",
    page_icon="🤱",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load("final_model_v2.joblib")
    artifacts     = joblib.load("app_artifacts.joblib")
    feature_names = joblib.load("feature_names.joblib")
    return model, artifacts, feature_names

try:
    model, artifacts, feature_names = load_model()
    screening_thr = artifacts["screening_thr"]
    highrisk_thr  = artifacts["highrisk_thr"]
    model_loaded  = True
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    model_loaded  = False

if model_loaded:
    st.title("🤱 Maternity Complications Risk Predictor")
    st.warning("⚠️ Decision support only. All clinical decisions remain with the clinician.")
    st.divider()

    st.subheader("📋 Patient Measurements")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Anthropometric**")
        height = st.number_input("Height (cm)",                 min_value=100, max_value=220, value=162)
        w1     = st.number_input("Weight — 1st Visit (kg)",     min_value=30,  max_value=200, value=68)
        w3     = st.number_input("Weight — 3rd Trimester (kg)", min_value=30,  max_value=200, value=78)
        st.markdown("**Obstetric History**")
        prev_prg = st.number_input("Previous Pregnancies", min_value=0, max_value=20, value=0)
        prev_dlv = st.number_input("Previous Deliveries",  min_value=0, max_value=20, value=0)
        prev_cs  = st.number_input("Previous C-Sections",  min_value=0, max_value=10, value=0)
        parity   = st.number_input("Parity",               min_value=0, max_value=20, value=0)

    with col2:
        st.markdown("**PHQ-2 (Depression Screen)**")
        phq2_1st = st.slider("PHQ-2 — 1st Visit",     0, 6, 0)
        phq2_3rd = st.slider("PHQ-2 — 3rd Trimester", 0, 6, 0)
        st.markdown("**Wexner (Bowel Function)**")
        wexner_1st = st.slider("Wexner — 1st Visit",     0, 20, 0)
        wexner_3rd = st.slider("Wexner — 3rd Trimester", 0, 20, 0)
        st.markdown("**ICIQ (Urinary Function)**")
        iciq_1st = st.slider("ICIQ — 1st Visit",     0, 21, 0)
        iciq_3rd = st.slider("ICIQ — 3rd Trimester", 0, 21, 0)

    st.markdown("**EQ-5D (Quality of Life)**")
    col3, col4 = st.columns(2)
    with col3:
        eq5d_ht_1st  = st.slider("EQ-5D Health Today — 1st Visit",      0, 100, 80)
        eq5d_idx_1st = st.slider("EQ-5D Index — 1st Visit (x100)",    -59, 100, 80)
    with col4:
        eq5d_ht_3rd  = st.slider("EQ-5D Health Today — 3rd Trimester", 0, 100, 75)
        eq5d_idx_3rd = st.slider("EQ-5D Index — 3rd Trimester (x100)",-59, 100, 75)

    eq5d_idx_1st = eq5d_idx_1st / 100
    eq5d_idx_3rd = eq5d_idx_3rd / 100
    st.divider()

    if st.button("🔍 PREDICT RISK", use_container_width=True):
        bmi1 = w1 / ((height / 100) ** 2)
        bmi3 = w3 / ((height / 100) ** 2)
        wg   = w3 - w1

        eng = {
            "height_cm__first_visit": height,
            "weight_kg__first_visit": w1,
            "weight_kg__third_trimester": w3,
            "prev_c_sections__first_visit": prev_cs,
            "prev_deliveries__first_visit": prev_dlv,
            "prev_pregnancies__first_visit": prev_prg,
            "phq2_total__first_visit": phq2_1st,
            "phq2_total__third_trimester": phq2_3rd,
            "eq5d_index__first_visit": eq5d_idx_1st,
            "eq5d_index__third_trimester": eq5d_idx_3rd,
            "eq5d_3l_healthtoday__first_visit": eq5d_ht_1st,
            "eq5d_3l_healthtoday__third_trimester": eq5d_ht_3rd,
            "wexner_total__first_visit": wexner_1st,
            "wexner_total__third_trimester": wexner_3rd,
            "iciq_total__first_visit": iciq_1st,
            "iciq_total__third_trimester": iciq_3rd,
            "bmi_first_visit": bmi1,
            "bmi_third_trimester": bmi3,
            "weight_gain": wg,
            "weight_gain_pct": (wg / w1 * 100) if w1 > 0 else np.nan,
            "bmi_x_weight_gain": bmi1 * wg,
            "binary_history_csection": float(prev_cs > 0),
            "prev_csection_rate": prev_cs / prev_dlv if prev_dlv > 0 else np.nan,
            "is_primiparous": float(parity == 0),
            "pregnancy_loss_rate": max(0, min(1, (prev_prg - prev_dlv) / prev_prg)) if prev_prg > 0 else np.nan,
            "obese_at_booking": float(bmi1 >= 30),
            "underweight_at_booking": float(bmi1 < 18.5),
            "excessive_weight_gain": float(wg > 16),
            "phq2_change": phq2_3rd - phq2_1st,
            "eq5d_index_change": eq5d_idx_3rd - eq5d_idx_1st,
            "wexner_change": wexner_3rd - wexner_1st,
            "iciq_change": iciq_3rd - iciq_1st,
            "phq2_positive_1st": float(phq2_1st >= 3),
            "phq2_positive_3rd": float(phq2_3rd >= 3),
            "poor_health_1st": float(eq5d_ht_1st < 50),
            "poor_health_3rd": float(eq5d_ht_3rd < 50),
            "any_bowel_symptoms_1st": float(wexner_1st > 0),
            "any_bowel_symptoms_3rd": float(wexner_3rd > 0),
            "age_over_35": np.nan,
        }

        df_input = pd.DataFrame([eng])
        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = np.nan

        try:
            prob = float(model.predict_proba(df_input[feature_names])[0, 1])

            st.subheader("🎯 Risk Assessment")
            if prob >= highrisk_thr:
                st.error(f"🔴 **HIGH RISK** — Probability: **{prob:.1%}**\n\nAction: URGENT clinical review required.")
            elif prob >= screening_thr:
                st.warning(f"🟡 **ELEVATED RISK** — Probability: **{prob:.1%}**\n\nAction: Increase monitoring frequency.")
            else:
                st.success(f"🟢 **LOW RISK** — Probability: **{prob:.1%}**\n\nAction: Continue standard antenatal care.")

            st.subheader("📋 Clinical Summary")
            col5, col6 = st.columns(2)
            with col5:
                st.metric("BMI at Booking",   f"{bmi1:.1f}", delta="⚠️ Obese" if bmi1 >= 30 else "✅ Normal", delta_color="inverse" if bmi1 >= 30 else "normal")
                st.metric("Weight Gain",       f"{wg:.1f} kg", delta="⚠️ Excessive" if wg > 16 else "✅ Normal", delta_color="inverse" if wg > 16 else "normal")
                st.metric("C-Section History", "Yes" if prev_cs > 0 else "No")
            with col6:
                st.metric("Bowel Symptoms (3rd)", "Yes ⚠️" if wexner_3rd > 0 else "No ✅")
                st.metric("Depression Risk",       "Positive ⚠️" if phq2_3rd >= 3 else "Negative ✅")
                st.metric("Model Probability",     f"{prob:.3f}")

            st.caption(f"Screening threshold: {screening_thr:.3f}  |  High-risk threshold: {highrisk_thr:.3f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
