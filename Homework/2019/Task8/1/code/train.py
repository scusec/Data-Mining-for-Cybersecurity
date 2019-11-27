from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint

import cv2
import os
import pickle
import numpy as np

EPOCHS = 16
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


def read_dataset(path):
    file_list = os.listdir(path)
    dataset = []
    for file in file_list:
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        dataset.append((img, int(file[:4])))
    pickle.dump(dataset, open("dataset.pickle", "wb"))


def dataset_to_train_test(dataset):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(0, 40000):
        if i % 4 == 0:
            X_test.append(np.transpose(np.array([dataset[i][0]]), (1, 2, 0)))
            y_test.append(dataset[i][1])
        else:
            X_train.append(np.transpose(np.array([dataset[i][0]]), (1, 2, 0)))
            y_train.append(dataset[i][1])
    return np.array(X_train), np.array(X_test), y_train, y_test


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last', name='layer1_con1',
                 input_shape=(60, 100, 1)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last', name='layer1_con2'))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last', name='layer1_pool2'))

model.add(Dropout(0.20))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last', name='layer1_con3'))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last', name='layer1_pool3'))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10000, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorBoard = TensorBoard(log_dir='./log', write_images=1, histogram_freq=1, write_graph=1)
checkpoint = ModelCheckpoint(filepath='./cnn_model', monitor='val_acc', mode='auto', save_best_only=True)

dataset = pickle.load(open("dataset.pickle", "rb"))

X_train, X_test, y_train, y_test = dataset_to_train_test(dataset)

y_train = to_categorical(y_train, 10000)
y_test = to_categorical(y_test, 10000)

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, callbacks=[checkpoint, tensorBoard],
          validation_split=VALIDATION_SPLIT)

print(model.evaluate(X_test, y_test, verbose=1, batch_size=BATCH_SIZE))
