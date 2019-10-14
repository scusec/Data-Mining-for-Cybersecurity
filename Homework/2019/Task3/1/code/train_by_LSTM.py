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

normal_data = []
xss_data = []
with open("dmzo_nomal.csv", "r") as f:
    for line in f:
        normal_data.extend(line.split(";"))

with open("xssed.csv", "r") as f:
    for line in f:
        xss_data.extend(line.split(";"))

# print(len(normal_data))
# print(len(xss_data))

data = []
data.extend(normal_data)
data.extend(xss_data)

labels = []
labels.extend([0] * len(normal_data))
labels.extend([1] * len(xss_data))

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(data)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

token_path = 'tokenizer.pkl'
pickle.dump(tokenizer, open(token_path, 'wb'))

index = [i for i in range(len(data))]
random.shuffle(index)
data = np.array(data)[index]
labels = np.array(labels)[index]

TRAIN_SIZE = int(0.8 * len(data))

X_train, X_test = data[0:TRAIN_SIZE], data[TRAIN_SIZE:]
Y_train, Y_test = labels[0:TRAIN_SIZE], labels[TRAIN_SIZE:]

session = tf.Session()
K.set_session(session)

QA_EMBED_SIZE = 64
DROPOUT_RATE = 0.3

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH))
model.add(
    Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=False, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE)))
model.add(Dense(QA_EMBED_SIZE))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.summary()


EPOCHS = 2
BATCH_SIZE = 64 * 4
VALIDATION_SPLIT = 0.3

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(
    'model-blstm.h5', save_best_only=True, save_weights_only=False)
tensor_board = TensorBoard(
    'log/tflog-blstm', write_graph=True, write_images=True)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=VALIDATION_SPLIT, shuffle=True,
          callbacks=[early_stopping, model_checkpoint, tensor_board])

model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCH_SIZE)
