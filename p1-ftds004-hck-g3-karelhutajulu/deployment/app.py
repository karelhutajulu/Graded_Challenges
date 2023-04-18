import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
import json

#Load All Files
model_scaling = joblib.load('standard_scaler.pkl')
random_forest = joblib.load('rf_model_best.pkl')

with st.form(key='Patient Information'):
    # Define the form elements
    age = st.slider('Age (years)', min_value=21, max_value=115, value=60, step=1, help='Patient age')
    ejection_fraction = st.slider('Ejection Fraction (%)', min_value=0, max_value=100, help='Percentage of blood leaving the heart with each contraction')
    time = st.number_input('Follow-up Period (days)', min_value=1, max_value=300, value=10, step=1, help='Duration of follow-up period')
    st.markdown('---')

    diabetes_option = st.radio('Diabetes', ('Yes', 'No'), help='Does the patient have diabetes?')
    diabetes = 1 if diabetes_option == 'Yes' else 0
    anaemia_option = st.radio('Anaemia', ('Yes', 'No'), help='Does the patient have anemia?')
    anaemia = 1 if anaemia_option == 'Yes' else 0
    high_blood_pressure_option = st.radio('High Blood Pressure', ('Yes', 'No'), help='Does the patient have high blood pressure?')
    high_blood_pressure = 1 if high_blood_pressure_option == 'Yes' else 0
    smoking_option = st.radio('Smoking', ('Yes', 'No'), help='Does the patient smoke?')
    smoking = 1 if smoking_option == 'Yes' else 0
    sex_option = st.radio('Sex', ('Male', 'Female'), help='Gender')
    sex = 1 if sex_option == 'Yes' else 0

    st.markdown('---')

    platelets = st.number_input('Platelets', min_value=1, max_value=850000, value=10, step=1, help='kiloplatelets/mL')
    creatinine_phosphokinase = st.number_input('Creatinine_phosphokinase', min_value=1, max_value=8000, value=10, step=1, help='mcg/L')
    serum_sodium = st.number_input('serum_sodium', min_value=1, max_value=300, value=10, step=1, help='mEq/dL')
    serum_creatinine = st.number_input('serum_creatinine', min_value=1, max_value=20, value=10, step=1, help='mg/dL')
    
    st.markdown('---')

    # Add a submit button to the form
    submitted = st.form_submit_button('Predict')

data_inf = {
    'age': age,
    'anaemia': anaemia,
    'creatinine_phosphokinase': creatinine_phosphokinase,
    'diabetes': diabetes,
    'ejection_fraction': ejection_fraction,
    'high_blood_pressure': high_blood_pressure,
    'platelets': platelets,
    'serum_creatinine': serum_creatinine,
    'serum_sodium': serum_sodium,
    'sex': sex,
    'smoking': smoking,
    'time': time
}

data_inf = pd.DataFrame([data_inf])
st.dataframe(data_inf)

if submitted:

    data_inf_features = data_inf[['age', 'ejection_fraction', 'time']]
    
    # Scale the selected columns using the pre-trained scaler
    scaled_data_inf_features = model_scaling.transform(data_inf_features)

    # Predict using the pre-trained random forest model
    predictions = random_forest.predict(scaled_data_inf_features)

    # Display the predictions using st.write
    st.write('Death Prediction:')
    # Display the predictions using st.write
    if predictions == 0:
        st.write('<p style="font-size:50px;">Alive</p>', unsafe_allow_html=True)
    elif predictions == 1:
        st.write('<p style="font-size:50px;">Dead</p>', unsafe_allow_html=True)


