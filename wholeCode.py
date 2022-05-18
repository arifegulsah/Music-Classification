# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:52:55 2022

@author: arife
"""

#import libraries
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

csvfilepath = 'csvfile/audioDatas.csv'

file = open(csvfilepath, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'genres/{g}'):
        songname = f'genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open(csvfilepath, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            

data = pd.read_csv(csvfilepath)
data.head()

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
data.head()

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
print(y)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()


history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)


test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)



predictions = model.predict(X_test)
np.argmax(predictions[2])

maxpredicts = []
for element in predictions:
    temp = np.argmax(element)
    maxpredicts.append(temp)
    

y_pred = np.array(maxpredicts)

  
"""
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(model, X_train, y_train)
"""


uniques = np.unique(y)
print(uniques)

target_names = ['0blues', '1classical', '2country', '3disco', '4hiphop', '5jazz', '6metal', '7pop', '8reggae', '9rock']

from sklearn.metrics import classification_report
classification_report(y_test, y_pred, target_names=target_names)

print(type(y_test))
print(type(y_pred))


