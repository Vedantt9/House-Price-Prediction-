# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:19:48 2022

@author: Hp
"""

import numpy as np 
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


loaded_model = pickle.load(open('F:/project/trained_model.sav', 'rb'))

train = pd.read_csv('Housee.csv')

le = preprocessing.LabelEncoder()
for name in train.columns:
    if train[name].dtypes == 'O':
        train[name] = train[name].astype(str)
        le.fit(train[name])
        train[name] = le.transform(train[name])
        
for column in train.columns:
    null_vals = train.isnull().values
    a, b = np.unique(train.values[~null_vals], return_counts = 1)
    train.loc[train[column].isna(), column] = np.random.choice(a, train[column].isnull().sum(), p = b / b.sum())
        
X = train.drop(['SalePrice', 'Id'], axis = 1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)
model.score(X_test, y_test)
back = np.expm1(model.predict(X_test[:1]))
back
