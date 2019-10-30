
import numpy as np
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import sklearn
from sklearn.model_selection import train_test_split
import mydata as data
import tensorflow as tf

def build_model(max_features, maxlen):
		net = tflearn.input_data(shape=[None,maxlen])
		
		
		net = tflearn.embedding(net, input_dim=maxlen, output_dim=512)
		
		net = tflearn.lstm(net, 128, dropout=0.5)
		net = tflearn.dropout(net, 0.5)
		
		net = tflearn.fully_connected(net, 1, activation='sigmoid')

		
		
		net = tflearn.regression(net, optimizer='rmsprop', learning_rate=0.001,
		loss='binary_crossentropy')
		
		# Training
		model = tflearn.DNN(net, tensorboard_verbose=0)
		
		return model


def run(max_epoch=25, nfolds=10, batch_size=128):
		"""Run train/test on logistic regression model"""
		indata = list(data.get_data())
		np.random.shuffle(indata)
		# Extract data and labels
		X = [x[1] for x in indata]
		labels = [x[0] for x in indata]

		# Generate a dictionary of valid characters
		valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

		max_features = len(valid_chars) + 1
		maxlen = np.max([len(x) for x in X])
		X = X[:2000]
		labels = labels[:2000]
		
		# Convert characters to int and pad
		X = [[valid_chars[y] for y in x] for x in X]
		X = pad_sequences(X, maxlen=maxlen,value=0.)

		# Convert labels to 0-1
		y = [0 if x == 'benign' else 1 for x in labels]

		
		for fold in range(nfolds):
			print ("fold %u/%u" % (fold+1, nfolds))
			X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, 
			                                                                   test_size=0.2)

			y_train = np.expand_dims(y_train, axis=-1)
			y_test = np.expand_dims(y_test, axis=-1)
			print('Build model...')
			model = build_model(max_features, maxlen)

			print( "Train...")

			for ep in range(max_epoch):
				model.fit(X_train, y_train, batch_size=batch_size,show_metric=True)

run()
