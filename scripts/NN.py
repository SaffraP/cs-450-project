#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:06:26 2020

@author: cannonbray
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#%%

dat = pd.read_csv("/Users/cannonbray/Documents/CS 450/CS450_group_project/document_term_matrix.csv")

#%%
#%%
dat = dat.dropna()

Y= pd.get_dummies(dat.data_target).values
X = dat.drop('data_target', axis = 1).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

#%%
### Build the model

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


model = Sequential() 
model.add(Dense(10000, input_dim=18694, activation= 'relu')) 
model.add(Dense(8000, activation = 'relu'))
model.add(Dense(6000, activation = 'relu'))
model.add(Dense(4000, activation = 'relu'))
model.add(Dense(2000, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dense(100, activation = 'relu')) 
model.add(Dense(2, activation = 'sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=['accuracy'])
# maxed out at about 0.58
#%%

history = model.fit(X_train, Y_train, epochs=5, batch_size=500, validation_split = 0.33)

#%%

model = Sequential() 
model.add(Dense(10000, input_dim=18694, activation= 'relu')) 
model.add(Dense(10000, activation = 'relu'))
model.add(Dense(10000, activation = 'relu'))
model.add(Dense(10000, activation = 'relu'))
model.add(Dense(10000, activation = 'relu'))
model.add(Dense(10000, activation = 'relu'))
model.add(Dense(10000, activation = 'relu')) 
model.add(Dense(10000, activation = 'relu'))
model.add(Dense(10000, activation = 'relu')) 
model.add(Dense(2, activation = 'sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=['accuracy'])
#%%

history = model.fit(X_train, Y_train, epochs=10, batch_size=2134, validation_split = 0.33)
#%%

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training accuracy', 'validation accuracy'], loc = "upper center", bbox_to_anchor = (0.5, -0.1), ncol = 2)
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training loss', 'validation loss'], loc = "upper center", bbox_to_anchor = (0.5, -0.1), ncol = 2)
plt.show()




