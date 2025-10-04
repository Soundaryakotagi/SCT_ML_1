import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("best_house_price_model.pkl")

# Streamlit UI
st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")

st.title("🏡 House Price Predictor")
st.write("""
Welcome! Enter the details of your house below to get an instant estimated market price.
""")

grliv_area = st.number_input(
    "🏠 Total Living Area (GrLivArea in sq.ft)",
    min_value=300, max_value=10000, value=1500,
    help="Above-ground living area. Typical: 800–4000 sq.ft."
)

bedrooms = st.number_input(
    "🛏 Number of Bedrooms (BedroomAbvGr)",
    min_value=0, max_value=10, value=3,
    help="Bedrooms above ground. Typical: 1–6"
)

full_bath = st.number_input(
    "🛁 Number of Full Bathrooms (FullBath)",
    min_value=0, max_value=10, value=2,
    help="Full bathrooms. Typical: 1–4"
)

if st.button("🔮 Predict Sale Price"):
    input_data = pd.DataFrame([{
        'GrLivArea': grliv_area,
        'BedroomAbvGr': bedrooms,
        'FullBath': full_bath
    }])

    prediction = model.predict(input_data)[0]
    prediction = max(prediction, 0)  # No negative values

    st.success(f"Estimated Sale Price: ₹{prediction:,.2f}")
