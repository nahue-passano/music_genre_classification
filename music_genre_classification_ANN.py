# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 02:55:32 2021

@author: NPass
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# =============================================================================
# Dataset process
# =============================================================================
def dataset_process(dataset_path):

    df = pd.read_csv(dataset_path)
    df = df.drop(labels = 'filename', axis=1)   # Drop the filename column
    
        # Encoding the genre labels into numerical values
    genre_list = df.iloc[:,-1]
    genre_encoder = LabelEncoder()
    targets_genre = genre_encoder.fit_transform(genre_list)
    
        # Scaling the features
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(np.array(df.iloc[:,:-1], dtype = float))
    
        # Split of the dataset into train and test sets
    X_train, X_test, target_train, target_test = train_test_split(dataset_scaled,targets_genre, test_size = 0.3)
    
    class dataset():
        class train():
            X = X_train
            target = target_train
        class test():
            X = X_test
            target = target_test
    
    
    return dataset()

# =============================================================================
# ANN build
# =============================================================================

def build_model(input_shape):
    
    model = Sequential()
    
    # 1st hidden layer
    model.add(Dense(units = 512 , activation = "relu" , input_shape = (input_shape,))),
    keras.layers.Dropout(0.2),
    
    # 2nd hidden layer
    model.add(Dense(units = 256 , activation = "relu" )),
    keras.layers.Dropout(0.2),
    
    # 3rd hidden layer
    model.add(Dense(units = 128 , activation = "relu" )),
    keras.layers.Dropout(0.2),
    
    # 4th hidden layer
    model.add(Dense(units = 64 , activation = "relu" )),
    keras.layers.Dropout(0.2),
    
    # Output layer
    model.add(Dense(units = 10 , activation = "softmax"))
    
    return model
    

# =============================================================================
# ANN train
# =============================================================================

def train_model(model, dataset, epochs, optimizer, batch_size = 128):
    
    model.compile(optimizer = optimizer, 
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")
    
    return model.fit(dataset.train.X, dataset.train.target,
                              validation_data = (dataset.test.X, dataset.test.target),
                              epochs = epochs,
                              batch_size = batch_size)
    

# =============================================================================
# ANN test
# =============================================================================

def test_model(model, dataset, batch_size = 128):
    
    test_loss, test_acc = model.evaluate(dataset.test.X, dataset.test.target, 
                                         batch_size = batch_size)
    
    return test_loss, test_acc
    
    
# =============================================================================
# Results
# =============================================================================

    # Dataset process
dataset_path = "features_3_sec.csv"
dataset = dataset_process(dataset_path)

    # ANN build
input_shape = dataset.train.X.shape[1] 
model = build_model(input_shape)
    
    # ANN train
epochs = 600
optimizer = "Adam"
train_model(model , dataset , epochs = epochs, optimizer = optimizer)

    # ANN test
test_loss, test_acc = test_model(model,dataset)
    
    
    
    
    
    
    
    
    
