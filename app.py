import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Patch custom function ─────────────────────────────────────────────────────
def to_str_array(x):
    x = pd.DataFrame(x).astype(object)
    x = x.where(~x.isna(), "__MISSING__")
    return x.astype(str).values

import sys
sys.modules["__main__"].to_str_array = to_str_array

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maternity Risk Predictor",
    page_icon="🤱",
    layout="wide"
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
    st.markdown("Enter **all** patient measurements for an accurate prediction.")
    st.warning("⚠️ Decision support only. All clinical decisions remain with the clinician.")
    st.divider()

    # =========================================================================
    # SECTION 1 — ANTHROPOMETRIC & OBSTETRIC
    # =========================================================================
    st.subheader("📏 Anthropometric & Obstetric History")
    col1, col2, col3 = st.columns(3)

    with col1:
        height   = st.number_input("Height (cm)",                   min_value=100, max_value=220, value=162)
        w1       = st.number_input("Weight — 1st Visit (kg)",       min_value=30,  max_value=200, value=68)
        w3       = st.number_input("Weight — 3rd Trimester (kg)",   min_value=30,  max_value=200, value=78)

    with col2:
        prev_prg = st.number_input("Previous Pregnancies",          min_value=0, max_value=20, value=0)
        prev_dlv = st.number_input("Previous Deliveries",           min_value=0, max_value=20, value=0)
        prev_cs  = st.number_input("Previous C-Sections",           min_value=0, max_value=10, value=0)

    with col3:
        multiple_gest = st.selectbox("Multiple Gestation",          ["No", "Yes"])
        num_fetuses   = st.number_input("Number of Fetuses",        min_value=1, max_value=6,  value=1)
        race_eth      = st.selectbox("Race / Ethnicity", [
            "White", "Black or African American", "Asian",
            "Hispanic or Latino", "Mixed", "Other", "Prefer not to say"
        ])
        edu_level     = st.selectbox("Education Level", [
            "Primary", "Secondary", "University", "Postgraduate", "Other"
        ])

    st.divider()

    # =========================================================================
    # SECTION 2 — EQ-5D (Quality of Life)
    # =========================================================================
    st.subheader("🌡️ EQ-5D Quality of Life Scores")
    st.markdown("Rate each dimension: **1** = No problems, **2** = Some problems, **3** = Extreme problems")

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**First Visit**")
        eq5d_mo_1st  = st.slider("Mobility (MO) — 1st Visit",             1, 3, 1)
        eq5d_sc_1st  = st.slider("Self-Care (SC) — 1st Visit",            1, 3, 1)
        eq5d_ua_1st  = st.slider("Usual Activities (UA) — 1st Visit",     1, 3, 1)
        eq5d_pd_1st  = st.slider("Pain/Discomfort (PD) — 1st Visit",      1, 3, 1)
        eq5d_ad_1st  = st.slider("Anxiety/Depression (AD) — 1st Visit",   1, 3, 1)
        eq5d_ht_1st  = st.slider("Health Today (0-100) — 1st Visit",      0, 100, 80)
        eq5d_idx_1st = st.slider("EQ-5D Index — 1st Visit (×100)",      -59, 100, 80)

    with col5:
        st.markdown("**Third Trimester**")
        eq5d_mo_3rd  = st.slider("Mobility (MO) — 3rd Trimester",         1, 3, 1)
        eq5d_sc_3rd  = st.slider("Self-Care (SC) — 3rd Trimester",        1, 3, 1)
        eq5d_ua_3rd  = st.slider("Usual Activities (UA) — 3rd Trimester", 1, 3, 1)
        eq5d_pd_3rd  = st.slider("Pain/Discomfort (PD) — 3rd Trimester",  1, 3, 1)
        eq5d_ad_3rd  = st.slider("Anxiety/Depression (AD) — 3rd Trimester", 1, 3, 1)
        eq5d_ht_3rd  = st.slider("Health Today (0-100) — 3rd Trimester",  0, 100, 75)
        eq5d_idx_3rd = st.slider("EQ-5D Index — 3rd Trimester (×100)",  -59, 100, 75)

    eq5d_idx_1st = eq5d_idx_1st / 100
    eq5d_idx_3rd = eq5d_idx_3rd / 100

    st.divider()

    # =========================================================================
    # SECTION 3 — PROM SCORES
    # =========================================================================
    st.subheader("📋 Patient-Reported Outcome Scores")
    col6, col7 = st.columns(2)

    with col6:
        st.markdown("**First Visit**")
        phq2_1st    = st.slider("PHQ-2 Depression Score — 1st Visit (0-6)",      0, 6,  0)
        wexner_1st  = st.slider("Wexner Bowel Score — 1st Visit (0-20)",         0, 20, 0)
        iciq_1st    = st.slider("ICIQ Urinary Score — 1st Visit (0-21)",         0, 21, 0)
        simss_1st   = st.slider("SIMSS Score — 1st Visit (0-40)",                0, 40, 0)
        sexfs_1st   = st.slider("SexFS Score — 1st Visit (0-40)",                0, 40, 0)

    with col7:
        st.markdown("**Third Trimester**")
        phq2_3rd    = st.slider("PHQ-2 Depression Score — 3rd Trimester (0-6)",  0, 6,  0)
        wexner_3rd  = st.slider("Wexner Bowel Score — 3rd Trimester (0-20)",     0, 20, 0)
        iciq_3rd    = st.slider("ICIQ Urinary Score — 3rd Trimester (0-21)",     0, 21, 0)

    st.divider()

    # =========================================================================
    # PREDICT
    # =========================================================================
    if st.button("🔍 PREDICT RISK", use_container_width=True):

        # ── Derive engineered features ────────────────────────────────────────
        bmi1 = w1 / ((height / 100) ** 2)
        bmi3 = w3 / ((height / 100) ** 2)
        wg   = w3 - w1

        eng = {
            # Raw clinical
            "height_cm__first_visit":               height,
            "weight_kg__first_visit":               w1,
            "weight_kg__third_trimester":           w3,
            "prev_c_sections__first_visit":         prev_cs,
            "prev_deliveries__first_visit":         prev_dlv,
            "prev_pregnancies__first_visit":        prev_prg,
            "multiple_gestation__third_trimester":  1.0 if multiple_gest == "Yes" else 0.0,
            "num_fetuses__third_trimester":         float(num_fetuses),
            "race_ethnicity__first_visit":          race_eth.lower(),
            "education_level__first_visit":         edu_level.lower(),

            # EQ-5D
            "eq5d_mo__first_visit":                 float(eq5d_mo_1st),
            "eq5d_sc__first_visit":                 float(eq5d_sc_1st),
            "eq5d_ua__first_visit":                 float(eq5d_ua_1st),
            "eq5d_pd__first_visit":                 float(eq5d_pd_1st),
            "eq5d_ad__first_visit":                 float(eq5d_ad_1st),
            "eq5d_3l_healthtoday__first_visit":     float(eq5d_ht_1st),
            "eq5d_index__first_visit":              eq5d_idx_1st,
            "eq5d_mo__third_trimester":             float(eq5d_mo_3rd),
            "eq5d_sc__third_trimester":             float(eq5d_sc_3rd),
            "eq5d_ua__third_trimester":             float(eq5d_ua_3rd),
            "eq5d_pd__third_trimester":             float(eq5d_pd_3rd),
            "eq5d_ad__third_trimester":             float(eq5d_ad_3rd),
            "eq5d_3l_healthtoday__third_trimester": float(eq5d_ht_3rd),
            "eq5d_index__third_trimester":          eq5d_idx_3rd,

            # PROMs
            "phq2_total__first_visit":              float(phq2_1st),
            "phq2_total__third_trimester":          float(phq2_3rd),
            "wexner_total__first_visit":            float(wexner_1st),
            "wexner_total__third_trimester":        float(wexner_3rd),
            "iciq_total__first_visit":              float(iciq_1st),
            "iciq_total__third_trimester":          float(iciq_3rd),
            "simss_total__first_visit":             float(simss_1st),
            "sexfs_total__first_visit":             float(sexfs_1st),

            # Engineered features
            "bmi_first_visit":                      bmi1,
            "bmi_third_trimester":                  bmi3,
            "weight_gain":                          wg,
            "weight_gain_pct":                      (wg / w1 * 100) if w1 > 0 else np.nan,
            "bmi_x_weight_gain":                    bmi1 * wg,
            "binary_history_csection":              float(prev_cs > 0),
            "prev_csection_rate":                   prev_cs / prev_dlv if prev_dlv > 0 else np.nan,
            "pregnancy_loss_rate":                  max(0, min(1, (prev_prg - prev_dlv) / prev_prg)) if prev_prg > 0 else np.nan,
            "obese_at_booking":                     float(bmi1 >= 30),
            "underweight_at_booking":               float(bmi1 < 18.5),
            "excessive_weight_gain":                float(wg > 16),
            "phq2_change":                          float(phq2_3rd - phq2_1st),
            "eq5d_index_change":                    eq5d_idx_3rd - eq5d_idx_1st,
            "wexner_change":                        float(wexner_3rd - wexner_1st),
            "iciq_change":                          float(iciq_3rd - iciq_1st),
            "phq2_positive_1st":                    float(phq2_1st >= 3),
            "phq2_positive_3rd":                    float(phq2_3rd >= 3),
            "poor_health_1st":                      float(eq5d_ht_1st < 50),
            "poor_health_3rd":                      float(eq5d_ht_3rd < 50),
            "any_bowel_symptoms_1st":               float(wexner_1st > 0),
            "any_bowel_symptoms_3rd":               float(wexner_3rd > 0),
        }

        # ── Build input dataframe ─────────────────────────────────────────────
        df_input = pd.DataFrame([eng])
        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = np.nan

        try:
            prob = float(model.predict_proba(df_input[feature_names])[0, 1])

            # ── Risk result ───────────────────────────────────────────────────
            st.subheader("🎯 Risk Assessment")
            if prob >= highrisk_thr:
                st.error(
                    f"🔴 **HIGH RISK** — Probability: **{prob:.1%}**\n\n"
                    f"Action: **URGENT** clinical review required before delivery."
                )
            elif prob >= screening_thr:
                st.warning(
                    f"🟡 **ELEVATED RISK** — Probability: **{prob:.1%}**\n\n"
                    f"Action: Increase monitoring frequency."
                )
            else:
                st.success(
                    f"🟢 **LOW RISK** — Probability: **{prob:.1%}**\n\n"
                    f"Action: Continue standard antenatal care."
                )

            # ── Clinical summary ──────────────────────────────────────────────
            st.subheader("📋 Clinical Summary")
            col8, col9, col10 = st.columns(3)

            with col8:
                st.metric("BMI at Booking",       f"{bmi1:.1f}",
                          delta="⚠️ Obese" if bmi1 >= 30 else "✅ Normal",
                          delta_color="inverse" if bmi1 >= 30 else "normal")
                st.metric("BMI 3rd Trimester",    f"{bmi3:.1f}")
                st.metric("Weight Gain",          f"{wg:.1f} kg",
                          delta="⚠️ Excessive" if wg > 16 else "✅ Normal",
                          delta_color="inverse" if wg > 16 else "normal")

            with col9:
                st.metric("Bowel Symptoms (1st)", "Yes ⚠️" if wexner_1st > 0 else "No ✅")
                st.metric("Bowel Symptoms (3rd)", "Yes ⚠️" if wexner_3rd > 0 else "No ✅")
                st.metric("Depression Risk",      "Positive ⚠️" if phq2_3rd >= 3 else "Negative ✅")

            with col10:
                st.metric("C-Section History",   "Yes" if prev_cs > 0 else "No")
                st.metric("Multiple Gestation",  multiple_gest)
                st.metric("Model Probability",   f"{prob:.3f}")

            st.divider()
            st.caption(
                f"Screening threshold: {screening_thr:.3f}  |  "
                f"High-risk threshold: {highrisk_thr:.3f}  |  "
                f"Features used: {len(feature_names)}"
            )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
