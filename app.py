# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Load trained model
# model = joblib.load("rental_price_model.pkl")

# # App title
# st.title("Dubai Rental Price Prediction")
# st.write("Fill the details below and click Predict")

# # User inputs
# beds = st.number_input("Bedrooms", value=2, min_value=0)
# baths = st.number_input("Bathrooms", value=2, min_value=0)
# area = st.number_input("Area (sqft)", value=800, min_value=100)

# # Optional inputs with default values
# property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Townhouse"])  # default choice
# rent_per_sqft = st.number_input("Rent per sqft", value=50)
# rent_category = st.selectbox("Rent Category", ["Low", "Medium", "High"])
# furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
# age_of_listing_in_days = st.number_input("Age of Listing (days)", value=10)
# location = st.text_input("Location", value="Dubai Marina")
# city = st.text_input("City", value="Dubai")
# latitude = st.number_input("Latitude", value=25.2048)
# longitude = st.number_input("Longitude", value=55.2708)

# # Predict button
# if st.button("Predict Rent"):
#     # Create dataframe in the same order as training
#     input_df = pd.DataFrame([[beds, baths, property_type, area, rent_per_sqft,
#                               rent_category, furnishing, age_of_listing_in_days,
#                               location, city, latitude, longitude]],
#                             columns=['Beds', 'Baths', 'Type', 'Area_in_sqft', 'Rent_per_sqft',
#                                      'Rent_category', 'Furnishing', 'Age_of_listing_in_days',
#                                      'Location', 'City', 'Latitude', 'Longitude'])
    
#     prediction = model.predict(input_df)
#     st.success(f"Estimated Rent: AED {int(prediction[0])}")


import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("rental_price_model_git.pkl")

st.title("🏠 Rental Price Prediction")
st.write("Enter numeric values exactly as used during training")

# ==============================
# Numeric Inputs (NO STRINGS)
# ==============================
beds = st.number_input("Beds", min_value=0, max_value=10, value=2)
baths = st.number_input("Baths", min_value=0, max_value=10, value=2)

property_type = st.number_input(
    "Type (numeric encoded)",
    help="Use same numeric encoding as training",
    value=1
)

area = st.number_input("Area in sqft", value=800)

rent_sqft = st.number_input("Rent per sqft", value=30.0)

rent_category = st.number_input(
    "Rent Category (numeric encoded)",
    help="Low=0, Medium=1, High=2 (example)",
    value=1
)

furnishing = st.number_input(
    "Furnishing (numeric encoded)",
    help="Unfurnished=0, Semi=1, Fully=2 (example)",
    value=1
)

age_days = st.number_input("Age of listing (days)", value=30)

location = st.number_input(
    "Location (numeric encoded)",
    help="Use same location code as training",
    value=5
)

city = st.number_input(
    "City (numeric encoded)",
    help="Use same city code as training",
    value=0
)

latitude = st.number_input("Latitude", value=25.2048)
longitude = st.number_input("Longitude", value=55.2708)

# ==============================
# Prediction
# ==============================
if st.button("Predict Rent 💰"):
    input_data = np.array([[ 
        beds,
        baths,
        property_type,
        area,
        rent_sqft,
        rent_category,
        furnishing,
        age_days,
        location,
        city,
        latitude,
        longitude
    ]])

    prediction = model.predict(input_data)
    st.success(f"🏷️ Estimated Rent: AED {int(prediction[0])}")
