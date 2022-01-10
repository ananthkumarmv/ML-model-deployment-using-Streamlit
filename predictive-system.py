# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 22:43:13 2022

@author: Unbeknownstguy
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('traied_model.sav', 'rb'))

input_data = (6,148,72,35,0,33.6,0.627,50)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


# predicting
prediction = loaded_model.predict(input_data_reshaped)

if(prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
    
    