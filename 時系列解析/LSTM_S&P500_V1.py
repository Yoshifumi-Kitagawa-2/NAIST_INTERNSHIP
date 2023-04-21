#!/usr/bin/env python3

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# データセットの読み込み
df = pd.read_csv('S&P500_V1.csv')

# 日付列をインデックスに設定
df = df.set_index('日付け')

# 終値だけを抽出
df = df[['終値']]

# データセットの先頭5行を表示
print(df.head())

# データのスケーリング
scaler = MinMaxScaler(feature_range=(0, 1))
df['終値'] = df['終値'].str.replace(',', '').astype(float)
df['終値'] = scaler.fit_transform(df['終値'].values.reshape(-1,1))

# 訓練データとテストデータに分割
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

# 訓練データとテストデータを確認
print(len(train), len(test))

# 時系列データに変換する関数の定義
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# 訓練データを時系列データに変換
dataset = df.values
time_step = 100
X_train, y_train = create_dataset(dataset, time_step)

# LSTMモデルの構築
model = Sequential()
## LSTMレイヤーをモデルに追加します。50個のLSTMユニットを含み、return_sequences=Trueに設定されているため、出力が時系列データの場合に次のLSTMレイヤーに渡されます。input_shape=(time_step, 1)は、モデルの入力形状を定義します。ここでは、各入力シーケンスがtime_step個の時間ステップと1つの特徴量（価格）を持つことを意味します。
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
## さらに1つのLSTMレイヤーを追加します。50個のLSTMユニットを含み、return_sequences=Trueに設定されているため、出力が時系列データの場合に次のLSTMレイヤーに渡されます。
model.add(LSTM(50, return_sequences=True))
## さらに1つのLSTMレイヤーを追加します。50個のLSTMユニットを含み、return_sequences=Falseに設定されているため、最終的な出力は1次元のベクトルになります。
model.add(LSTM(50))
## 密結合層を追加します。1つのノードを持ち、最終的な出力を生成します。
model.add(Dense(1))
## モデルをコンパイルします。損失関数として平均二乗誤差を使用し、オプティマイザーとしてAdamを使用します。
model.compile(loss='mean_squared_error', optimizer='adam')

# モデルの学習
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# テストデータを時系列データに変換
X_test, y_test = create_dataset(test.values, time_step)

# テストデータを予測
y_pred = model.predict(X_test)

# 予測結果のスケーリングを元に戻す
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

# 結果のプロット
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.gca().invert_xaxis()
plt.legend()
plt.show()
