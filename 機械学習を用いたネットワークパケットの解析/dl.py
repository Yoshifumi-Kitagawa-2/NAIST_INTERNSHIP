#!/usr/bin/env python3

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_AFFINITY'] = 'disabled'

from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import optuna
from optuna import Trial, visualization

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

df = pd.read_csv('agg_packets.csv')
df_tmp=df.loc[:,["packets_count","TCP_count","UDP_count","ICMP_count","PA_count","FPA_count","S_count","SA_count","A_count","FA_count","IP_src_count","IP_dst_count","IP_sport_count","IP_dport_count","ddos_flag"]]
values=df_tmp.values

scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

x_scaled=values_scaled[:,0:-1]
y_scaled=values_scaled[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
y_train_c = keras.utils.to_categorical(y_train, 2)
y_test_c = keras.utils.to_categorical(y_test, 2)

from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_train1, y_train1 = smote.fit_resample(x_train, y_train)
y_train1_c = keras.utils.to_categorical(y_train1, 2)

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1)

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(14,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train1, y_train1_c,
          epochs=100,
          batch_size=32,
          callbacks=[early_stop],
          verbose=2,
          validation_data=(x_test, y_test_c))

predicted = np.argmax(model.predict(x_test, verbose=0), axis=-1)
expected = y_test
from sklearn import metrics
print(metrics.confusion_matrix(expected, predicted))
print(metrics.classification_report(expected, predicted))