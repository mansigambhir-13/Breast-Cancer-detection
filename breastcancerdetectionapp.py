import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit app
st.title('Seer Breast Cancer Prediction')

st.write("""
This app predicts the likelihood of breast cancer based on the Seer dataset.
Please enter the required information below.
""")

# Create input fields for all features
features = {}

# Mean features
features['radius_mean'] = st.slider('Radius (mean)', 0.0, 30.0, 15.0, 0.1)
features['texture_mean'] = st.slider('Texture (mean)', 0.0, 40.0, 20.0, 0.1)
features['perimeter_mean'] = st.slider('Perimeter (mean)', 0.0, 200.0, 100.0, 0.1)
features['area_mean'] = st.slider('Area (mean)', 0.0, 2500.0, 1000.0, 1.0)
features['smoothness_mean'] = st.slider('Smoothness (mean)', 0.0, 0.2, 0.1, 0.001)
features['compactness_mean'] = st.slider('Compactness (mean)', 0.0, 0.3, 0.1, 0.001)
features['concavity_mean'] = st.slider('Concavity (mean)', 0.0, 0.5, 0.2, 0.001)
features['concave points_mean'] = st.slider('Concave points (mean)', 0.0, 0.2, 0.1, 0.001)
features['symmetry_mean'] = st.slider('Symmetry (mean)', 0.0, 0.3, 0.2, 0.001)
features['fractal_dimension_mean'] = st.slider('Fractal dimension (mean)', 0.0, 0.1, 0.05, 0.001)

# SE (Standard Error) features
features['radius_se'] = st.slider('Radius (SE)', 0.0, 2.0, 0.5, 0.01)
features['texture_se'] = st.slider('Texture (SE)', 0.0, 5.0, 1.0, 0.01)
features['perimeter_se'] = st.slider('Perimeter (SE)', 0.0, 20.0, 5.0, 0.1)
features['area_se'] = st.slider('Area (SE)', 0.0, 150.0, 50.0, 1.0)
features['smoothness_se'] = st.slider('Smoothness (SE)', 0.0, 0.02, 0.005, 0.0001)
features['compactness_se'] = st.slider('Compactness (SE)', 0.0, 0.05, 0.02, 0.0001)
features['concavity_se'] = st.slider('Concavity (SE)', 0.0, 0.1, 0.03, 0.0001)
features['concave points_se'] = st.slider('Concave points (SE)', 0.0, 0.02, 0.01, 0.0001)
features['symmetry_se'] = st.slider('Symmetry (SE)', 0.0, 0.03, 0.02, 0.0001)
features['fractal_dimension_se'] = st.slider('Fractal dimension (SE)', 0.0, 0.01, 0.003, 0.0001)

# Worst features
features['radius_worst'] = st.slider('Radius (worst)', 0.0, 40.0, 20.0, 0.1)
features['texture_worst'] = st.slider('Texture (worst)', 0.0, 50.0, 25.0, 0.1)
features['perimeter_worst'] = st.slider('Perimeter (worst)', 0.0, 250.0, 125.0, 0.1)
features['area_worst'] = st.slider('Area (worst)', 0.0, 4000.0, 2000.0, 1.0)
features['smoothness_worst'] = st.slider('Smoothness (worst)', 0.0, 0.3, 0.15, 0.001)
features['compactness_worst'] = st.slider('Compactness (worst)', 0.0, 1.0, 0.5, 0.001)
features['concavity_worst'] = st.slider('Concavity (worst)', 0.0, 1.0, 0.5, 0.001)
features['concave points_worst'] = st.slider('Concave points (worst)', 0.0, 0.3, 0.15, 0.001)
features['symmetry_worst'] = st.slider('Symmetry (worst)', 0.0, 0.5, 0.25, 0.001)
features['fractal_dimension_worst'] = st.slider('Fractal dimension (worst)', 0.0, 0.2, 0.1, 0.001)

# Convert features to DataFrame
input_df = pd.DataFrame([features])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader('Prediction')
    st.write('Cancer Likelihood:', 'Malignant' if prediction[0] == 1 else 'Benign')
    
    st.subheader('Prediction Probability')
    st.write(f'Probability of Benign: {prediction_proba[0][0]:.2f}')
    st.write(f'Probability of Malignant: {prediction_proba[0][1]:.2f}')

st.write("""
Note: This is a demonstration and should not be used for actual medical diagnosis. 
Always consult with a healthcare professional for medical advice.
""")
