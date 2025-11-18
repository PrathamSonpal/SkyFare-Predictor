# app.py - Flight Fare Prediction Streamlit App
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="centered")

st.title("✈️ SkyFare Predictor")
st.write("Provide flight details below to predict the ticket price (INR).")

@st.cache_resource
def load_model():
    with open("SkyFare-Predictor.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Try to extract learned categories from pipeline (safe fallback to lists)
def get_category_list(step_index):
    try:
        ohe = model.named_steps["transform"].transformers_[0][1]
        cats = ohe.categories_[step_index].tolist()
        return sorted([str(c) for c in cats if pd.notna(c)])
    except Exception:
        return None

airlines = get_category_list(0) or ["IndiGo", "Air India", "Vistara", "SpiceJet", "GO FIRST"]
source_cities = get_category_list(1) or ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Hyderabad", "Chennai"]
destination_cities = get_category_list(2) or source_cities
travel_classes = ["economy", "business"]

st.subheader("Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", airlines)
    source_city = st.selectbox("Source City", source_cities)
    travel_class = st.selectbox("Class", travel_classes)
    dep_hour = st.slider("Departure Hour", 0, 23, 10)
    dep_min = st.slider("Departure Minute", 0, 59, 0)

with col2:
    destination_city = st.selectbox("Destination City", destination_cities)
    stops_num = st.number_input("Number of Stops", min_value=0, max_value=4, value=0)
    duration_mins = st.number_input("Duration (minutes)", min_value=10, max_value=2000, value=120, step=5)
    arr_hour = st.slider("Arrival Hour", 0, 23, 12)
    arr_min = st.slider("Arrival Minute", 0, 59, 0)

days_left = st.number_input("Days Left for Journey", 0, 365, 20)

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
        predicted_price = model.predict(input_df)[0]
        st.success(f"Estimated Ticket Price: ₹ {predicted_price:,.2f}")
    except Exception as e:
        st.error("Prediction failed. Check that the model file and feature names match.")
        st.exception(e)
