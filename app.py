
import streamlit as st
import pandas as pd, numpy as np, joblib, os, glob, traceback
from PIL import Image
st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥", layout="wide")

# --- Helpers ---
def load_model():
    MODEL_DIR = "models"
    load_errors = {}
    loaded_model = None
    loaded_model_name = None
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib"))):
        name = os.path.basename(path)
        try:
            m = joblib.load(path)
            if hasattr(m, "predict") or hasattr(m, "predict_proba"):
                loaded_model = m
                loaded_model_name = name
                break
        except Exception as e:
            load_errors[name] = str(e)
    return loaded_model, loaded_model_name, load_errors

model, model_name, load_errors = load_model()
if model is None:
    st.error("No compatible model could be loaded. See load errors below.")
    for k,v in load_errors.items():
        st.write(f"- **{k}**: {v[:400]}")
    st.stop()

st.sidebar.image(Image.new("RGB", (1,1), (11,99,206)), width=1)  # tiny color bar for style
st.sidebar.title("🏥 Hospital Tools")
st.sidebar.write("Model loaded:")
st.sidebar.write(f"**{model_name}**")

st.title("Hospital Readmission Predictor")
st.markdown("Predict the likelihood of patient readmission. Use **Single Patient** for one-off checks or **Batch Scoring** to score a CSV.")

# Attempt to infer feature names
feature_names = None
if hasattr(model, "feature_names_in_"):
    try:
        feature_names = list(model.feature_names_in_)
    except Exception:
        feature_names = None

# If pipeline with ColumnTransformer, try to extract columns
from sklearn.pipeline import Pipeline
try:
    if isinstance(model, Pipeline):
        for name, step in model.named_steps.items():
            from sklearn.compose import ColumnTransformer
            if isinstance(step, ColumnTransformer):
                cols_try = []
                for tname, trans, cols in step.transformers:
                    if isinstance(cols, (list, tuple)):
                        cols_try.extend(list(cols))
                if cols_try:
                    feature_names = cols_try
except Exception:
    pass

# Best-effort human-readable mappings for common encoded feature names
readable_map = {
    "age": "Age Range",
    "time_in_hospital": "Time in Hospital (days)",
    "num_procedures": "Number of Procedures",
    "num_medications": "Number of Medications",
    "number_outpatient_log": "Number of Outpatient Visits (log)",
    "number_emergency_log": "Number of Emergency Visits (log)",
    "number_inpatient_log": "Number of Inpatient Visits (log)",
    "number_diagnoses": "Number of Diagnoses",
    "max_glu_serum_1": "Max Glucose Serum: Normal",
    "max_glu_serum_99": "Max Glucose Serum: No Test",
    "A1Cresult_1": "A1C Result: Normal",
    "A1Cresult_99": "A1C Result: No Test",
    # medication binaries will be shown as Yes/No
}

# Define grouped sections for UI ordering
demographics = ["age", "gender_1", "race_1", "race_2", "race_3", "race_4"]
history = ["time_in_hospital", "num_medications", "num_procedures", "number_outpatient_log", "number_emergency_log", "number_inpatient_log", "number_diagnoses"]
labs = ["max_glu_serum_1", "max_glu_serum_99", "A1Cresult_1", "A1Cresult_99"]
diagnosis = [c for c in (feature_names or []) if c.startswith("primary_diag_")] if feature_names else []
meds = [c for c in (feature_names or []) if c in ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","glipizide","glyburide","pioglitazone","rosiglitazone","acarbose","tolazamide","insulin","glyburide-metformin"]]

# Tabs
tab1, tab2 = st.tabs(["🧍 Single Patient", "🗂️ Batch Scoring"])

with tab1:
    st.subheader("Patient details")
    with st.expander("Patient Demographics", expanded=True):
        cols = st.columns([2,2,2])
        demo_vals = {}
        for i, fname in enumerate(demographics):
            if fname in (feature_names or []):
                label = readable_map.get(fname, fname.replace("_"," ").title())
                if fname.startswith("race"):
                    demo_vals[fname] = st.selectbox(label, options=["Unknown","Caucasian","AfricanAmerican","Asian","Other"], index=0, key=fname)
                elif fname == "gender_1":
                    demo_vals[fname] = st.selectbox(label, options=["Male","Female","Other"], index=0, key=fname)
                else:
                    demo_vals[fname] = st.text_input(label, key=fname, value="30-40")
    with st.expander("Medical History", expanded=False):
        cols = st.columns([1,1,1])
        hist_vals = {}
        for i, fname in enumerate(history):
            if fname in (feature_names or []):
                label = readable_map.get(fname, fname.replace("_"," ").title())
                # numeric sliders for counts
                hist_vals[fname] = st.number_input(label, value=1.0, step=1.0, key=fname)
    with st.expander("Lab & Diagnosis", expanded=False):
        cols = st.columns([2,2,2])
        lab_vals = {}
        for fname in labs + diagnosis:
            if fname in (feature_names or []):
                label = readable_map.get(fname, fname.replace("_"," ").title())
                if fname in ["max_glu_serum_1","A1Cresult_1"]:
                    lab_vals[fname] = st.selectbox(label, options=["No","Normal","High","No Test"], key=fname)
                else:
                    lab_vals[fname] = st.text_input(label, key=fname, value="0")
    if meds:
        with st.expander("Medications", expanded=False):
            med_cols = st.columns(3)
            med_vals = {}
            for i, fname in enumerate(meds):
                if fname in (feature_names or []):
                    label = fname.replace("-"," ").title()
                    med_vals[fname] = st.selectbox(label, options=["No","Steady","Up","Down"], key=fname)
    # Combine values
    input_vals = {}
    for d in (demo_vals if 'demo_vals' in locals() else {}):
        input_vals[d] = demo_vals[d]
    for h in (hist_vals if 'hist_vals' in locals() else {}):
        input_vals[h] = hist_vals[h]
    for l in (lab_vals if 'lab_vals' in locals() else {}):
        input_vals[l] = lab_vals[l]
    for m in (med_vals if 'med_vals' in locals() else {}):
        input_vals[m] = med_vals[m]
    st.write("")

    if st.button("Predict Readmission", type="primary"):
        try:
            X = pd.DataFrame([input_vals])
            # basic casting: attempt numeric conversion
            for c in X.columns:
                try:
                    X[c] = pd.to_numeric(X[c])
                except Exception:
                    pass
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[:,1][0]
                pred = int(prob >= 0.5)
                # Styled result card
                if prob >= 0.5:
                    st.markdown(f"<div style='background:#FFEBEB;padding:16px;border-radius:8px'> <h3 style='color:#B00020'>High risk of readmission</h3><p>Probability: <b>{prob:.2%}</b></p></div>", unsafe_allow_html=True)
                    st.info("Recommendation: Consider close follow-up and discharge planning.")
                else:
                    st.markdown(f"<div style='background:#ECF8F1;padding:16px;border-radius:8px'> <h3 style='color:#116530'>Low risk of readmission</h3><p>Probability: <b>{prob:.2%}</b></p></div>", unsafe_allow_html=True)
                    st.success("Recommendation: Standard discharge process.")
            else:
                p = model.predict(X)[0]
                st.write("Prediction:", p)
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            st.text(traceback.format_exc())

with tab2:
    st.subheader("Batch CSV Scoring")
    st.write("Upload a CSV matching the model's training features. Use the sample template if needed.")
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("Upload CSV for scoring", type=["csv"])
    with col2:
        if os.path.exists("data/sample_batch.csv"):
            with open("data/sample_batch.csv", "rb") as f:
                st.download_button("Download sample template", f, "sample_batch.csv", "text/csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(50))
            # If feature_names available, check missing
            if feature_names:
                missing = [c for c in feature_names if c not in df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    X = df[feature_names]
            else:
                X = df
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:,1]
                preds = (probs >= 0.5).astype(int)
                out = df.copy()
                out["readmission_probability"] = probs
                out["prediction"] = np.where(preds==1, "Readmitted", "Not Readmitted")
            else:
                preds = model.predict(X)
                out = df.copy()
                out["prediction"] = preds
            st.success(f"Scored {len(out)} rows.")
            st.dataframe(out.head(100))
            st.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
        except Exception as e:
            st.error("Batch scoring failed: " + str(e))
            st.text(traceback.format_exc())

st.write("---")
st.caption("Built for demonstration. For production, secure the app and validate input thoroughly.")
