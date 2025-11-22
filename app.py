import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="SkyFare Predictor",
    page_icon="✈️",
    layout="centered"
)

# ------------------ TITLE & DESCRIPTION ------------------ #
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }
    .stApp {
        background-color: #f5f7fb;
    }
    .box {
        padding: 1rem 1.2rem;
        background-color: #ffffff;
        border-radius: 0.8rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.03);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">✈️ SkyFare Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict airline ticket prices instantly using Machine Learning</div>',
    unsafe_allow_html=True
)

# ------------------ LOAD MODEL ------------------ #
@st.cache_resource
def load_model():
    fname = "flight_fare_model.pkl"  # make sure this exact file exists in repo root
    try:
        return joblib.load(fname)
    except Exception as e_joblib:
        try:
            import pickle
            with open(fname, "rb") as f:
                return pickle.load(f)
        except Exception as e_pickle:
            st.error("Failed to load model. Please check model file name and format.")
            raise RuntimeError(f"joblib error: {e_joblib}\n\npickle error: {e_pickle}")

model = load_model()

# ------------------ CATEGORY OPTIONS FROM MODEL ------------------ #
def get_categories():
    try:
        ohe = model.named_steps["transform"].transformers_[0][1]
        cats = [list(map(str, c)) for c in ohe.categories_]
        return cats
    except Exception:
        return None

cats = get_categories()
airlines = cats[0] if cats else ["IndiGo", "Air India", "Vistara", "SpiceJet", "GO FIRST"]
sources  = cats[1] if cats else ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Hyderabad", "Chennai"]
dests_all = cats[2] if cats else sources
travel_classes = ["economy", "business"]

# ------------------ INPUT UI ------------------ #

st.markdown('<div class="section-title">🛫 Route & Class Details</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox("Airline", airlines)

    with col2:
        source_city = st.selectbox("Source City", sources)

    with col3:
        # dynamically filter destination so it can't be the same as source
        dest_options = [d for d in dests_all if d != source_city]
        if not dest_options:
            # fallback (shouldn't really happen, but safe)
            dest_options = dests_all
        destination_city = st.selectbox("Destination City", dest_options)

    col4, col5 = st.columns(2)
    with col4:
        travel_class = st.selectbox("Class", travel_classes)
    with col5:
        stops_num = st.number_input("Number of Stops", min_value=0, max_value=4, value=0, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">⏱ Time & Duration</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="box">', unsafe_allow_html=True)
    col_time1, col_time2 = st.columns(2)

    with col_time1:
        dep_hour = st.slider("Departure Hour", 0, 23, 10)
        dep_min = st.slider("Departure Minute", 0, 59, 0)
    with col_time2:
        arr_hour = st.slider("Arrival Hour", 0, 23, 12)
        arr_min = st.slider("Arrival Minute", 0, 59, 0)

    st.markdown("**Flight Duration**")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        duration_hours = st.number_input("Duration Hours", min_value=0, max_value=30, value=2, step=1)
    with col_d2:
        duration_minutes = st.number_input("Duration Minutes", min_value=0, max_value=59, value=0, step=5)

    duration_mins = int(duration_hours) * 60 + int(duration_minutes)
    st.caption(f"Total duration: {duration_hours} h {duration_minutes} min  →  {duration_mins} minutes")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">📅 Booking Details</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="box">', unsafe_allow_html=True)
    days_left = st.number_input("Days left for journey", min_value=0, max_value=365, value=15, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PREDICTION ------------------ #

st.markdown('<div class="section-title">🎯 Prediction</div>', unsafe_allow_html=True)

predict_button = st.button("Predict Ticket Price")

if predict_button:
    # Validation: prevent same source and destination
    if source_city == destination_city:
        st.error("Source and destination cities cannot be the same. Please select different cities.")
    else:
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
            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.subheader("💰 Estimated Ticket Price")
            st.markdown(f"<h3>₹ {pred:,.2f}</h3>", unsafe_allow_html=True)
            st.caption("Note: This is an estimated price based on historical patterns.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("Prediction failed. Please check model and input processing.")
            st.exception(e)

# ------------------ FOOTER ------------------ #
st.write("")
st.caption("Project: SkyFare Predictor · Built with Python, Scikit-Learn & Streamlit")
