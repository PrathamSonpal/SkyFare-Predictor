# debug_app.py (replace your app.py with this for debugging)
import streamlit as st
import os
import pandas as pd
import traceback

st.set_page_config(page_title="SkyFare Predictor", layout="centered")
st.title("✈️ SkyFare Predictor")
st.write("This debug page helps identify why the app shows a blank screen. Follow the messages below.")

# show environment info
st.subheader("Environment & Files")
try:
    st.write("Python executable:", os.sys.executable)
    st.write("Current working dir:", os.getcwd())
    files = sorted(os.listdir("."))
    st.write("Files in repo root (top 50):", files[:50])
except Exception as e:
    st.error("Failed to list files")
    st.exception(e)

# model load attempt with detailed errors
st.subheader("Model load check")
MODEL_NAMES = ["SkyFare-Predictor.pkl"]

model = None
model_name_used = None
load_errors = []

for fname in MODEL_NAMES:
    if os.path.exists(fname):
        st.write(f"Found candidate model file: `{fname}` (size: {os.path.getsize(fname)/1024/1024:.2f} MB)")
    else:
        st.write(f"Model not found: `{fname}`")

# Try to load with joblib then pickle, show exceptions
import importlib
try:
    joblib = importlib.import_module("joblib")
except Exception:
    joblib = None
    st.warning("joblib not available in runtime (will try pickle).")

for fname in MODEL_NAMES:
    if not os.path.exists(fname):
        continue
    try:
        if joblib:
            st.write(f"Trying joblib.load('{fname}') ...")
            model = joblib.load(fname)
            model_name_used = fname
            st.success(f"Loaded model with joblib from {fname}")
            break
    except Exception as e:
        st.error(f"joblib.load failed for {fname}")
        st.text(traceback.format_exc())
        load_errors.append(("joblib", fname, traceback.format_exc()))
    try:
        st.write(f"Trying pickle.load('{fname}') ...")
        import pickle
        with open(fname, "rb") as f:
            model = pickle.load(f)
        model_name_used = fname
        st.success(f"Loaded model with pickle from {fname}")
        break
    except Exception as e:
        st.error(f"pickle.load failed for {fname}")
        st.text(traceback.format_exc())
        load_errors.append(("pickle", fname, traceback.format_exc()))

if model is None:
    st.error("Model could not be loaded. See above errors.")
    st.markdown("**Next steps:**")
    st.markdown("- Check that the model file name (case-sensitive) in the repo root matches one of the names above.")
    st.markdown("- If you used `joblib.dump(..., compress=...)`, load with `joblib.load`.")
    st.markdown("- If the model file is huge (>25MB) GitHub may not have stored it; use a smaller model or host externally.")
    st.markdown("- Paste the full exception texts into the chat and I will fix them.")
else:
    st.success(f"Model is ready from: {model_name_used}")
    # quick smoke test predict (build a default input using model.feature_names_in_ if present)
    st.subheader("Quick smoke-test prediction")
    try:
        # create a safe sample depending on what model expects
        # try to get column names if transformer exists
        sample = None
        if hasattr(model, "named_steps"):
            # pipeline: find transformer category names and create a dummy row
            # fallback: try to get feature names from model if available
            # We'll create a simple sample using defaults
            sample = pd.DataFrame([{
                "airline": "IndiGo",
                "source_city": "Delhi",
                "destination_city": "Mumbai",
                "travel_class": "economy",
                "stops_num": 0,
                "duration_mins": 120,
                "dep_hour": 9,
                "dep_min": 30,
                "arr_hour": 11,
                "arr_min": 30,
                "days_left": 15
            }])
        else:
            # if raw model, skip
            sample = None

        if sample is not None:
            st.write("Sample input used for prediction:")
            st.write(sample)
            pred = model.predict(sample)
            st.success(f"Sample prediction: {pred}")
        else:
            st.write("No sample prediction attempted (model type unknown).")
    except Exception:
        st.error("Sample prediction failed. See trace below:")
        st.text(traceback.format_exc())

# show captured load errors succinctly for copy/paste
if load_errors:
    st.subheader("Captured load errors (first two shown)")
    for method, fname, tb in load_errors[:2]:
        st.markdown(f"**{method} failed for {fname}:**")
        st.code(tb[:4000])  # show first part

st.info("If you see errors above, copy the full traceback and paste here. I'll tell you the exact fix.")
