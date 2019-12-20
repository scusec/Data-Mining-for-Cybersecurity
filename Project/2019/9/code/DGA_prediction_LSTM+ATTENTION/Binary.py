from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import load_model
import numpy as np

def train(max_features, X_train, y_train, batch_size, epochs, modelPath):
	model=Sequential()
	model.add(Embedding(max_features,128,input_length=75)) # 词向量维度128，输入序列长75
	model.add(LSTM(128))
	model.add(Dropout(0.5)) # 概率为一般为0.5和0.3，防止过拟合
	model.add(Dense(1)) # 全连接神经网络层
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='rmsprop') # 交叉熵损失函数
	X_train=sequence.pad_sequences(X_train,maxlen=75)
	model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
	model.save(modelPath)


def predict(X_test, batch_size, modelPath, resultPath):
	X_test = sequence.pad_sequences(X_test, maxlen=75)
	my_model = load_model(modelPath)
	y_test = my_model.predict(X_test, batch_size=batch_size).tolist()
	file = open(resultPath, 'w+')
	for index in y_test:
		y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
		file.write(str(y) + '\n')
