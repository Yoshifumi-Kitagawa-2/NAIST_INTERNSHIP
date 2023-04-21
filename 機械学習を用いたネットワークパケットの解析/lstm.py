#!/usr/bin/env python3

#!/usr/bin/env python3

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# データセットの読み込み
df = pd.read_csv("agg_packets.csv")
df_tmp=df.loc[:,["packets_count","TCP_count","UDP_count","ICMP_count","PA_count","FPA_count","S_count","SA_count","A_count","FA_count","IP_src_count","IP_dst_count","IP_sport_count","IP_dport_count","ddos_flag"]]
values=df_tmp.values

# データのスケーリング
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values)

# 時系列データに変換する関数の定義
def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :-1]
        dataX.append(a)
        dataY.append(dataset[i + time_step + 1, -1])
    return np.array(dataX), np.array(dataY)

# 訓練データを時系列データに変換
dataset = values_scaled
time_step = 20
x_time_series, y_time_series= create_dataset(dataset, time_step)

# 時系列データをtrainとtest
x_train, x_test, y_train, y_test = train_test_split(x_time_series, y_time_series, test_size=0.2, random_state=42)
y_train_c = keras.utils.to_categorical(y_train, 2)
y_test_c = keras.utils.to_categorical(y_test, 2)

# LSTMモデルの構築
model = Sequential()
## LSTMレイヤーをモデルに追加します。50個のLSTMユニットを含み、return_sequences=Trueに設定されているため、出力が時系列データの場合に次のLSTMレイヤーに渡されます。input_shape=(time_step, 1)は、モデルの入力形状を定義します。ここでは、各入力シーケンスがtime_step個の時間ステップと1つの特徴量（価格）を持つことを意味します。
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, x_time_series.shape[2])))
## さらに1つのLSTMレイヤーを追加します。50個のLSTMユニットを含み、return_sequences=Trueに設定されているため、出力が時系列データの場合に次のLSTMレイヤーに渡されます。
model.add(LSTM(50, return_sequences=True))
## さらに1つのLSTMレイヤーを追加します。50個のLSTMユニットを含み、return_sequences=Falseに設定されているため、最終的な出力は1次元のベクトルになります。
model.add(LSTM(50))
## 密結合層を追加します。1つのノードを持ち、最終的な出力を生成します。
model.add(Dense(2, activation='softmax'))
## モデルをコンパイルします。損失関数として平均二乗誤差を使用し、オプティマイザーとしてAdamを使用します。
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルの学習
model.fit(x_train, y_train_c, validation_data=(x_test, y_test_c), epochs=30, batch_size=64, verbose=1)

# 結果のプロット
predicted = np.argmax(model.predict(x_test, verbose=0), axis=-1)
expected = y_test
from sklearn import metrics
print(metrics.confusion_matrix(expected, predicted))
print(metrics.classification_report(expected, predicted))
