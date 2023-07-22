import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import loan_data

# A predictive model (SVC) to predict the loan status of an applicant.
@st.cache_data
def get_fvalue(val):
    feature_dict = {"No":0, "Yes":1}
    for key, value in feature_dict.items():
        if val == key:
            return value
        
def get_value(val, my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = "Prediction"
classifier = loan_data.preprocess_data()

if app_mode=="Prediction":
    st.title("Fill out your information on the left")
    st.image("loan_pic.jpg")
    gender_dict = {'Male':1,'Female':0}
    feature_dict = {"No":0,"Yes":1}
    edu = {"Graduate":1, "Not Graduate":0}
    property_dict = {"Rural": 0, "Urban": 1, "Semiurban":2}
    ApplicantIncome = st.sidebar.slider("Applicant Income", 0,10000,0,)
    CoapplicantIncome = st.sidebar.slider("Coapplicant Income", 0,10000,0,)
    Credit_History = st.sidebar.radio("Credit_History", (0.0,1.0))
    Gender = st.sidebar.radio("Gender", tuple(gender_dict.keys()))
    Married = st.sidebar.radio("Married", tuple(feature_dict.keys()))
    Education = st.sidebar.radio("Education", tuple(edu.keys()))
    Property_Area = st.sidebar.radio("Property_Area", tuple(property_dict.keys()))
    
    data1={'Gender':Gender,'Married':Married,
           'Education':Education,
           'ApplicantIncome':ApplicantIncome,
           'CoapplicantIncome':CoapplicantIncome,
           'Credit_History':Credit_History,'Property_Area':property_dict[Property_Area]}
    feature_list=[ApplicantIncome,CoapplicantIncome,Credit_History,
                  get_value(Gender,gender_dict),get_fvalue(Married),
                  get_value(Education,edu),data1['Property_Area']]
    single_sample = np.array(feature_list).reshape(1,-1)
    
    if st.sidebar.button("Predict"):
        single_sample = np.array(feature_list).reshape(1,-1)
        prediction = loan_data.predict_res(classifier, single_sample)
        if prediction[0] == 0 :
            st.error('According to our Calculations, you will not get the loan from Bank')
        elif prediction[0] == 1 :
            st.success('Congratulations!! you will get the loan from Bank')
            st.balloons()