# -*- coding:utf-8 -*-
# import os
# os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import load_model
import numpy as np



def train(max_features, nb_classes, X_train, y_train, batch_size, epochs, modelPath):

	model=Sequential()
	model.add(Embedding(max_features,128,input_length=75))
	model.add(LSTM(128))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

	X_train=sequence.pad_sequences(X_train,maxlen=75)
	model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
	model.save(modelPath)

def predict(X_test, batch_size, modelPath, resultPath):
	X_test = sequence.pad_sequences(X_test, maxlen=75)
	my_model = load_model(modelPath)
	y_test = my_model.predict(X_test, batch_size=batch_size).tolist()

	file = open(resultPath, 'w+')
	for y in y_test:
		max_index = 0
		max_num = 0.0

		for i in range(len(y)):
			if y[i] > max_num:
				max_index = i
				max_num = y[i]

		file.write(str(max_index) + '\n')