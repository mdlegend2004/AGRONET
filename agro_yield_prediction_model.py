import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load the trained model (for production prediction)
model = load_model('crop_yield_prediction_model.h5')

# Load the saved label encoders and scaler
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit appi 611

def main():
    st.title("Crop Production Prediction")

    # Get user inputs
    state = st.selectbox("Select State", options=label_encoders['State'].classes_)
    district = st.selectbox("Select District", options=label_encoders['District'].classes_)
    crop = st.selectbox("Select Crop", options=label_encoders['Crop'].classes_)
    season = st.selectbox("Select Season", options=label_encoders['Season'].classes_)
    area = st.number_input("Enter Area (in hectares)", min_value=0.0, step=0.1)

    if st.button("Predict Production"):
        # Encode categorical inputs
        state_encoded = label_encoders['State'].transform([state])[0]
        district_encoded = label_encoders['District'].transform([district])[0]
        crop_encoded = label_encoders['Crop'].transform([crop])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]
        area_units_encoded = label_encoders['Area Units'].transform(['Hectare'])[0]  # Assuming 'Hectare'
        
        # Production Units are no longer an input since production is the output
        
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'State': [state_encoded],
            'District': [district_encoded],
            'Crop': [crop_encoded],
            'Season': [season_encoded],
            'Area': [area],
            'Area Units': [area_units_encoded]
        })

        # Scale numerical inputs
        input_data[['Area']] = scaler.transform(input_data[['Area']])

        # Predict production
        production_prediction = model.predict(input_data)

        st.success(f"Predicted Production: {production_prediction[0][0]:.2f} tonnes")

if __name__ == '__main__':
    main()
