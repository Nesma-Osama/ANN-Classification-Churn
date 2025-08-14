import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
#load model
model=load_model('model.h5')
#scaler
with open("scaling.pkl","rb")as f:
    scaler=pickle.load(f)

#label encoder
with open("label_encoder_gender.pkl","rb")as f:
    label_encoder=pickle.load(f)

#one hot encoder
with open("one_encoder_geo.pkl","rb")as f:
    one_hot_encoder=pickle.load(f)
    
# streamlit app
st.title("Customer Churn Prediction")
age=st.slider("Age", 18, 99)
gender=st.selectbox("Gender",label_encoder.classes_)
geography=st.selectbox("Geography",one_hot_encoder.categories_[0]) 
credit_score=st.number_input("Credit Score")
tenure=st.number_input("Tenure", 0, 10)
balance=st.number_input("Balance")
no_of_products=st.number_input("Number of Products", 1, 4)
has_credit_card=st.selectbox("Has Credit Card", [0, 1])
is_active_member=st.selectbox("Is Active Member", [0, 1])
estimated_salary=st.number_input("Estimated Salary")
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Geography':[geography],
    'Gender':[gender],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[no_of_products],
    'HasCrCard':[has_credit_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
    
})

input_data['Gender']=label_encoder.transform(input_data['Gender'])
one_hot=pd.DataFrame(one_hot_encoder.transform([input_data['Geography']]),columns=one_hot_encoder.get_feature_names_out(['Geography']))
x_test=pd.concat((input_data.drop(['Geography'],axis=1),one_hot), axis=1)
x_test_scaled=scaler.transform(x_test)

prediction=model.predict(x_test_scaled)[0][0]
if prediction > 0.5:
    st.write("The customer is likely to churn.")   
else:
    st.write("The customer is not likely to churn.")