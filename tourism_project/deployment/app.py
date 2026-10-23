"""Streamlit app: collects customer inputs into a dataframe and predicts purchase."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "best_tourism_model.joblib"

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧘", layout="wide")
st.title("Wellness Tourism Package - Purchase Predictor")
st.write("Enter the customer's details to estimate whether they will buy the package.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run the GitHub Actions pipeline so it commits the trained model.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Customer")
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", 0, 200000, 20000, step=500)
with col2:
    st.subheader("Trip profile")
    city = st.selectbox("City Tier", [1, 2, 3])
    persons = st.number_input("Number of Persons Visiting", 1, 10, 3)
    kids = st.number_input("Children Visiting (under 5)", 0, 5, 0)
    star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    trips = st.number_input("Trips per Year", 0, 25, 3)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
    car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
with col3:
    st.subheader("Sales interaction")
    contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.number_input("Duration of Pitch (minutes)", 1, 150, 15)
    followups = st.number_input("Number of Follow-ups", 0, 10, 3)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# inputs -> dataframe (column names must match the training data)
input_df = pd.DataFrame([{
    "Age": age, "TypeofContact": contact, "CityTier": city, "DurationOfPitch": duration,
    "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups, "ProductPitched": product,
    "PreferredPropertyStar": star, "MaritalStatus": marital, "NumberOfTrips": trips,
    "Passport": passport, "PitchSatisfactionScore": pitch_score, "OwnCar": car,
    "NumberOfChildrenVisiting": kids, "Designation": designation, "MonthlyIncome": income,
}])

st.subheader("Input summary")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict", type="primary"):
    prob = float(model.predict_proba(input_df)[0, 1])
    if prob >= 0.5:
        st.success(f"Likely to purchase the Wellness Tourism Package (probability {prob:.1%})")
    else:
        st.warning(f"Unlikely to purchase the package (probability {prob:.1%})")
