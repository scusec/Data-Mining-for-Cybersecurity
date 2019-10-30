import pickle
import copy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

MAX_LENGTH = 46  # 域名长度不得超过46字节
EMBEDDING_DIM = 37  # 算上26个英文字母，10个数字和-，一共37种字符
TEST_RATIO = 0.2

non_dga = []
dga = []

with open("dga_processed.pickle", "rb") as f:
    dga = pickle.load(f)

with open("non_dga_processed.pickle", "rb") as f:
    non_dga = pickle.load(f)

non_dga = non_dga[:10000]

data = copy.deepcopy(dga)
data.extend(non_dga)

labels = [1] * len(dga)
labels.extend([0] * len(non_dga))

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data)
sequence = tokenizer.texts_to_sequences(data)
data = pad_sequences(sequence, maxlen=MAX_LENGTH)

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

DROPOUT_RATE = 0.3

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, input_length=MAX_LENGTH))
model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE)))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.summary()

EPOCHS = 16
BATCH_SIZE = 64 * 2
VALIDATION_SPLIT = 0.3

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('model-blstm.h5', save_best_only=True, save_weights_only=False)
tensor_board = TensorBoard('log/tflog-blstm', write_graph=True, write_images=True, write_grads=True,
                           update_freq='batch')


def recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    rec = tp / (pp + K.epsilon())
    return rec


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall])

model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=VALIDATION_SPLIT, shuffle=True,
          callbacks=[early_stopping, model_checkpoint, tensor_board])

print(model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCH_SIZE))
