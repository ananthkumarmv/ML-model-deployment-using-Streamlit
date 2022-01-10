# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 23:07:35 2022

@author: Unbeknownstguy
"""

import numpy as np
import pickle
import streamlit


# loading the saved model
loaded_model = pickle.load(open('traied_model.sav', 'rb'))


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
        
        


