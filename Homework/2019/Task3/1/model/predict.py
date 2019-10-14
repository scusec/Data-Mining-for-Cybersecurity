from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import random
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100

model = load_model("model-blstm.h5")

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

data=["lang=en"]

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(data)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

predict_test=model.predict(data)
for i in predict_test:
    if i[0]<0.5:
        print("不是XSS吧    "+str(i[0]))
    else:
        print("是XSS吧    "+str(i[0]))
