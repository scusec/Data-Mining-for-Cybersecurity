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


def recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    rec = tp / (pp + K.epsilon())
    return rec


MAX_SEQUENCE_LENGTH = 46
EMBEDDING_DIM = 37

model = load_model("model-blstm.h5", custom_objects={"recall": recall})

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

data = ["example.com"]

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(data)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

predict_test = model.predict(data)
for i in predict_test:
    if i[0] < 0.5:
        print("不是DGA域名吧    " + str(i[0]))
    else:
        print("是DGA域名吧    " + str(i[0]))
