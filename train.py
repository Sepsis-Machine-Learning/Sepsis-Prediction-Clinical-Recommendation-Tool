import tensorflow as tf
from tensorflow import keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

train = pd.read_csv("setB_train.psv", sep='|')
#cols = [7]
# print(train.shape)
train.HospAdmTime = -1 * train.HospAdmTime
#train=train.drop(train.columns[cols],axis=1)
train[np.isnan(train)] = -99
train=train.values

x_train_raw = train[:, :-1] # for all but last column
n_rows, n_cols = x_train_raw.shape

y_train = train[:, -1]  # for last column

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(n_cols,)),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])
keras.optimizers.Adam(lr = 0.001)
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

x_train = NormalizeData(x_train_raw)
model.fit(x_train, y_train, epochs=100, batch_size=200)
model.save("model1")