import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

@st.cache(allow_output_mutation=True)
def load(scaler_url, model_url):
    scaler = joblib.load(scaler_url)
    model = joblib.load(model_url)
    return scaler, model

def inference(row, scaler, model, columns):
    X = pd.DataFrame([row], columns=columns)
    X = scaler.transform(X)
    if(model.predict(X)==0): diagnosis = "Healthy"
    else: diagnosis = "Highly likely diabetic"
    return diagnosis

st.title('Diabetes Prediction App')
st.write('The data for the following example is originally from the National Institute of Diabetes and Digestive and Kidney Diseases and contains information on females at least 21 years old of Pima Indian heritage. This is a sample application and cannot be used as a substitute for real medical advice.')
# image = Image.open('data/diabetes_image.jpg')
# st.image(image, use_column_width=True)
st.write('Please fill in the details of the person under consideration in the left sidebar and click on the button below!')

age =           st.sidebar.number_input("Age in Years", 1, 150, 25, 1)
pregnancies =   st.sidebar.number_input("Number of Pregnancies", 0, 20, 0, 1)
glucose =       st.sidebar.slider("Glucose Level", 0, 200, 25, 1)
skinthickness = st.sidebar.slider("Skin Thickness", 0, 99, 20, 1)
bloodpressure = st.sidebar.slider('Blood Pressure', 0, 122, 69, 1)
insulin =       st.sidebar.slider("Insulin", 0, 846, 79, 1)
bmi =           st.sidebar.slider("BMI", 0.0, 67.1, 31.4, 0.1)
dpf =           st.sidebar.slider("Diabetics Pedigree Function", 0.000, 2.420, 0.471, 0.001)

row = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]

if (st.button('Find Health Status')):
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    scaler, model = load('models/scaler.joblib', 'models/rfc-model.joblib')
    result = inference(row, scaler, model, columns)
    st.write(result)