#!/usr/bin/python
# _*_ coding: utf-8 _*_

from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras.layers import Embedding, Dense, Dropout, LSTM
import numpy as np
import io
from sklearn.model_selection import train_test_split

#loading the parsed files
def load_data(file):
    with io.open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result

x_normal = load_data("normal_parsed.txt")
x_anomalous = load_data("anomalous_parsed.txt")

#Creating the dataset
x = x_normal  + x_anomalous

#creating labels normal=0, anomalous=1
y_normal = [0] * len(x_normal)
y_anomalous = [1] * len(x_anomalous)
y = y_normal + y_anomalous

#assigning indices to each character in the query string
tokenizer = Tokenizer(char_level=True) #treating each character as a token
tokenizer.fit_on_texts(x) #training the tokenizer on the text

#creating the numerical sequences by mapping the indices to the characters
sequences = tokenizer.texts_to_sequences(x)
char_index = tokenizer.word_index


#to see the list of characters with their indices:
print(char_index)

maxlen = 1000   #length of the longest sequence=input_length

#padding the sequences to the same length
x = pad_sequences(sequences, maxlen=maxlen)
y = np.asarray(y)

#shuffle the dataset since the samples are ordered (normal requests first then anomalous requests)
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

#spliting the dataset into train and test 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

#print(x_train.shape)
#print(x_test.shape)

#creating the validation set
x_val = x_train[:20000]
partial_x_train = x_train[20000:]
y_val = y_train[:20000]
partial_y_train = y_train[20000:]

#size of the vector space in which characters will be embedded
embedding_dim = 32

#size of the vocabulary or input_dim
max_chars = 63


def build_model():
  model = models.Sequential()
  model.add(layers.Embedding(max_chars, embedding_dim, input_length=maxlen))
  model.add(layers.LSTM(100))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1, activation='sigmoid'))
  model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])
  return model

model = build_model()
print(model.summary())

'''
#evaluating the model over 100 epochs 
model.fit(partial_x_train, partial_y_train, epochs=100, batch_size=32)
val_acc = model.evaluate(x_val, y_val)
print(val_acc) 
'''
#training the model on the entire training set and evaluating it using the testing data
model.fit(x_train, y_train, epochs=40, batch_size=32)
test_acc, test_loss = model.evaluate(x_test, y_test)
print(test_acc, test_loss)

