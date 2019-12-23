# %%

from keras.models import Model
from keras import layers
from keras import Input
import pandas as pd
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv("./Final.csv")

# %%

data.dropna(axis=0, how='any', inplace=True)

# %%

Fraud_target = data["isFraud"]
text = data["str"]
text = text.values
data.drop('str', axis=1, inplace=True)
data.drop('isFraud', axis=1, inplace=True)
num = data.values
# %%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_len = 30
word_max = 50
tokenizer = Tokenizer(num_words=word_max)

texts = ""
for i in text:
    texts += i
tokenizer.fit_on_texts(texts)
sequence = tokenizer.texts_to_sequences(text)
text_seq = pad_sequences(sequence, max_len)
# %%
original = []
for i in range(len(text)):
    original.append([num[i], text_seq[i]])

# %%
import numpy

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(original, Fraud_target, test_size=0.3, random_state=42)
# y_binary = y_train.to_numpy(dtype=numpy.int32)
target = Fraud_target.to_numpy(dtype=numpy.int32)
X_text = []
X_num = []
scaler = MinMaxScaler()
# for i in X_train:
#     X_train_num.append(i[0])
#     X_train_text.append(i[1])
# X_test_num = []
# X_test_text = []
# for i in X_test:
#     X_test_num.append(i[0])
#     X_test_text.append(i[1])
# for i in original:
#     X_num.append(i[0])
#     X_text.append(i[1])

# %%
# X_test_num=scaler.fit_transform(X_test_num)
# X_train_num=scaler.fit_transform(X_train_num)

from sklearn.model_selection import StratifiedKFold
import numpy

seed = 7
numpy.random.seed(seed)
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# X_num = scaler.fit_transform(X_num)

text_input = Input(shape=(30,), dtype="int32", name="text")
embedded_text = layers.Embedding(50, 64)(text_input)
text_dense = layers.Dense(64)(embedded_text)
text_dense2 = layers.Dense(32)(text_dense)
text_flatten = layers.Flatten()(text_dense2)

from keras.layers import Dense

float_input = Input(shape=(37,), dtype="float32", name="data")
dense1 = Dense(64, input_shape=(37,))(float_input)
dense2 = Dense(64, input_shape=(64,))(dense1)
dense3 = Dense(32, input_shape=(64,))(dense2)
concat = layers.concatenate([text_flatten, dense3], axis=1)
result = layers.Dense(1, activation="sigmoid")(concat)
model = Model([text_input, float_input], result)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
for train, test in kfold.split(original, target):
    train_text = []
    train_num = []
    test_text = []
    test_num = []
    for i in np.array(original)[train]:
        train_text.append(i[1])
        train_num.append(i[0])
    for i in np.array(original)[test]:
        test_num.append(i[0])
        test_text.append(i[1])
    train_num = scaler.fit_transform(train_num)
    test_num = scaler.fit_transform(test_num)
    model.fit([train_text, train_num], target[train], epochs=100, validation_data=[[test_text, test_num], target[test]],
              validation_split=0.1, batch_size=128)
