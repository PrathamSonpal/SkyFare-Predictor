# at top of app.py
import joblib
import streamlit as st
import pandas as pd

@st.cache_resource
def load_model():
    fname = "SkyFare-Predictor.pkl"   # or "flight_fare_model_small.pkl" if that's the exact name in your repo
    try:
        # prefer joblib because model was saved with joblib.dump (supports compress)
        model = joblib.load(fname)
        return model
    except Exception as e_joblib:
        # fallback to pickle for older models saved with pickle.dump
        import pickle
        try:
            with open(fname, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e_pickle:
            # show full error in Streamlit logs and raise to stop app
            st.error("Loading model failed. See app logs for details.")
            # combine exceptions to give more info in logs
            raise RuntimeError(f"joblib error: {e_joblib}\n\npickle error: {e_pickle}")
