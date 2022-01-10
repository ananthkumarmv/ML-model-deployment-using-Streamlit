# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 23:07:35 2022

@author: Unbeknownstguy
"""

import numpy as np
import pickle  # used for loading the saved model
import streamlit as st # used for creating the webpage


# loading the saved model
loaded_model = pickle.load(open('C:/Users/Unbeknownstguy/Documents/GitHub/Projects/Machine_Learning/ML-model-deployment-using-Streamlit/traied_model.sav', 'rb'))


# Creating a function for prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


    # predicting
    prediction = loaded_model.predict(input_data_reshaped)

    if(prediction[0]==0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
        
        

def main():
    
    
    # giviin a title
    st.title('Diabetes Preiction Web App')
    
    
    # getting the input data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('Body Mass Index Value') 
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
