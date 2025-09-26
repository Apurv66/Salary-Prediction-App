import streamlit as st
import pickle
import numpy as np
import pandas as pd

df = pd.read_csv("clean_data.csv")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Gender_encoder.pkl", "rb") as f:
    gender_le = pickle.load(f)
with open("Education Level_encoder.pkl", "rb") as f:
    edu_le = pickle.load(f)
with open("Job Title_encoder.pkl", "rb") as f:
    job_le = pickle.load(f)

st.header("Salary Prediction Model")
st.divider()
age = st.number_input("Enter Age: ", min_value=15, max_value=70, step=1)
gender = st.selectbox("Gender:", ['Male', 'Female'])
edu_level = st.selectbox("Education Level", df['Education Level'].unique())
job_title = st.selectbox("Job Title", df['Job Title'].unique())
year_exp = st.number_input("Years of Experience", min_value=0, max_value=70, step=1)

if st.button("Predict"):
    gender_val = gender_le.transform([gender])[0]
    edu_val = edu_le.transform([edu_level])[0]
    job_val = job_le.transform([job_title])[0]

    input_data = np.array([[age, gender_val, edu_val, job_val, year_exp]])

    pred = model.predict(input_data)
    
    st.success(f"Predicted Salary: ₹{pred[0]:.0f}")
