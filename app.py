# app.py - SkyFare Predictor (production)
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="SkyFare Predictor", page_icon="✈️", layout="centered")
st.title("✈️ SkyFare Predictor")
st.write("Predict airline ticket prices instantly using Machine Learning ✈️📊")

@st.cache_resource
def load_model():
    fname = "SkyFare-Predictor.pkl" 
    # try joblib then pickle
    try:
        return joblib.load(fname)
    except Exception as e_joblib:
        try:
            import pickle
            with open(fname, "rb") as f:
                return pickle.load(f)
        except Exception as e_pickle:
            st.error("Failed to load model. Check model file and logs.")
            raise RuntimeError(f"joblib error: {e_joblib}\n\npickle error: {e_pickle}")

model = load_model()

def get_categories():
    try:
        ohe = model.named_steps["transform"].transformers_[0][1]
        cats = [list(map(str, c)) for c in ohe.categories_]
        return cats
    except Exception:
        return None

cats = get_categories()
airlines = cats[0] if cats else ["IndiGo","Air India","Vistara","SpiceJet"]
sources  = cats[1] if cats else ["Delhi","Mumbai","Bengaluru"]
dests    = cats[2] if cats else sources

travel_classes = ["economy","business"]

col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox("Airline", airlines)
    source_city = st.selectbox("Source City", sources)
    travel_class = st.selectbox("Class", travel_classes)
    dep_hour = st.slider("Departure Hour", 0, 23, 10)
    dep_min  = st.slider("Departure Minute", 0, 59, 0)
with col2:
    destination_city = st.selectbox("Destination City", dests)
    stops_num = st.number_input("Number of Stops", min_value=0, max_value=4, value=0)
    duration_mins = st.number_input("Duration (minutes)", min_value=10, max_value=2000, value=120, step=5)
    arr_hour = st.slider("Arrival Hour", 0, 23, 12)
    arr_min  = st.slider("Arrival Minute", 0, 59, 0)

days_left = st.number_input("Days left for journey", min_value=0, max_value=365, value=15)

if st.button("Predict Ticket Price"):
    input_df = pd.DataFrame([{
        "airline": airline,
        "source_city": source_city,
        "destination_city": destination_city,
        "travel_class": travel_class,
        "stops_num": int(stops_num),
        "duration_mins": int(duration_mins),
        "dep_hour": int(dep_hour),
        "dep_min": int(dep_min),
        "arr_hour": int(arr_hour),
        "arr_min": int(arr_min),
        "days_left": int(days_left)
    }])
    try:
        pred = model.predict(input_df)[0]
        st.success(f"Estimated Ticket Price: ₹ {pred:,.2f}")
    except Exception as e:
        st.error("Prediction failed. Check model and feature names.")
        st.exception(e)
