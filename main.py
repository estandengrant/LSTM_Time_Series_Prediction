import random
import warnings
import keras.optimizers
warnings.simplefilter('ignore')
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import pandas as pd
from collections import deque


SEQUENCE_LEN = 60    # Historical period to use
FUTURE_PERIODS = 3  # Future datapoints we are trying to predict
EPOCHS = 3
BATCH = 64

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def process(df):

    # Get % change of the closing price and volume - this helps to standardize
    df.close = df.close.pct_change()
    df.volume = df.volume.pct_change()
    df.dropna(inplace=True)

    # Scale the closing price and volume data
    df.close = preprocessing.scale(df.close.values)
    df.volume = preprocessing.scale(df.volume.values)
    df.dropna(inplace=True)

    # Create sequences of length 60 that the model can use to predict future price movement.
    # Each sequence is associated with a target label (the value in column 'Target') - this label indicates the price movement 3 periods after the sequence finishes.
    seq_data = []
    prev_days = deque(maxlen=SEQUENCE_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQUENCE_LEN:
            seq_data.append([np.array(prev_days), i[-1]])
    random.shuffle(seq_data)

    # Separate data into two lists (Buys and Sells) based on their label. This is useful when assessing the balance of data.
    buys = []
    sells = []
    for seq, targets in seq_data:
        if targets == 0:
            sells.append([seq, targets])
        else:
            buys.append([seq, targets])
    random.shuffle(buys)
    random.shuffle(sells)

    # Assess the balance of the data and ensure we have equal amounts of each label.
    lesser_val = min(len(buys), len(sells))
    buys = buys[:lesser_val]
    sells = sells[:lesser_val]

    seq_data = buys+sells
    random.shuffle(seq_data)

    # Separate sequence and label into X and y lists
    X = []
    y = []
    for seq, targets in seq_data:
        X.append(seq)
        y.append(targets)

    return np.array(X), np.array(y)


df = pd.read_csv('DOGE_USD.csv')
df.set_index('time', inplace=True)
df = df[['close', 'volume']]
df['future'] = df.close.shift(-FUTURE_PERIODS) # I.e.the price at time t+3 will be df['future']
df['target'] = list(map(classify, df['close'], df['future']))
df = df.drop('future', axis=1)

# Split data into train and test sets - we use the final 15% of data to test
times = sorted(df.index.values)
last_15pct = times[-int(0.15*len(times))]
validation_df = df[df.index >= last_15pct]
df = df[df.index < last_15pct]

train_x, train_y = process(df)
validation_x, validation_y = process(validation_df)



# Build model
model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True, activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True, activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=False, activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=BATCH,epochs=EPOCHS,validation_data=(validation_x,validation_y))

