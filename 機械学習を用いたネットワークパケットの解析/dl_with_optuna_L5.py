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
from sklearn.metrics import f1_score
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

#early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1)

#from sklearn import metrics
#print(metrics.confusion_matrix(expected, predicted))
#print(metrics.classification_report(expected, predicted))

def objective(trial: Trial):
    # ハイパーパラメータの設定
    n_layers = 5
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_layer{i}', 10, 100))
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    batch = trial.suggest_int('batch', 1, 64)
    n_epochs = 100
    
    # モデルの定義
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(14,)))
    for i in range(1, n_layers):
        model.add(Dense(layers[i], activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax'))

    # モデルのコンパイル
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr), metrics=['accuracy'])

    # 学習
    model.fit(x_train1, y_train1_c, epochs=n_epochs, batch_size=batch, callbacks=[EarlyStopping(monitor='val_loss', patience=8, verbose=1)], validation_data=(x_test, y_test_c), verbose=2)

    # テストデータでの精度を計算
    predicted = np.argmax(model.predict(x_test, verbose=0), axis=-1)
    expected = y_test
    f1 = f1_score(expected, predicted)

    return f1

study = optuna.create_study(study_name='DDoS_detection_result_L5',
                                storage='sqlite:///DDoS_detection_result_L5.db',
                                load_if_exists=True,
                                direction='maximize')
study.optimize(objective, timeout=95*60*60)

# 最適なハイパーパラメータを出力
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value:.5f}')
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
   
# 最適化の過程を可視化
visualization.plot_optimization_history(study)

# ハイパーパラメータごとの重要度を可視化
visualization.plot_param_importances(study)