# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:08:13 2022

@author: Hp
"""

import numpy as np 
import pandas as pd
from math import sqrt
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import streamlit as st

loaded_model = pickle.load(open('F:/project/trained_model.sav', 'rb'))

def house_pred(input_data):
     
    back = np.expm1(input_data)
    back
    
  
    return 'Predicted Price'


def main():
    
    st.title('House price Prediction')
    
    
    GrLivArea = st.text_input(' Area')
    FullBath = st.number_input('No of floors')
    BedroomAbvGr = st.number_input('No of BHK')
    GarageArea = st.number_input('Car Parking')
    LotArea = st.number_input('washroom')
    Total = (FullBath+BedroomAbvGr+GarageArea+LotArea)
   
    
    
    
    tot = ''
    
    if st.button('Submit'):
        #tot = house_pred([GrLivArea,FullBath,BedroomAbvGr,GarageArea,LotArea])
        tot = house_pred([Total])
        
    st.success(tot)




if __name__ == '__main__':
    main()
    