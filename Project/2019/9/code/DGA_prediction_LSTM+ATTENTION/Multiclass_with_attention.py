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
from keras.layers import merge
from keras.layers.core import *
from keras.models import *


def linear_attention_global(inputs,TIME_STEPS):
    # inputs.shape = (batch_size, time_steps, input_dim)
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax', name='dense1')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def nonlinear_attention_relu(inputs,TIME_STEPS):
    # inputs.shape = (batch_size, time_steps, input_dim)
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a = Dense(30, activation='relu')(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def nonlinear_attention_tanh(inputs,TIME_STEPS):
    # inputs.shape = (batch_size, time_steps, input_dim)
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a = Dense(30, activation='tanh')(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def model_attention_applied_after_lstm(shape, max_features, nb_classes):
    inputs = Input(shape=(shape,))
    a = Embedding(max_features, 128, input_length=shape)(inputs)
    lstm_out = LSTM(128, return_sequences=True)(a)  #返回值 (time_steps, input_dim) = (75,128)


    # attention_mul = linear_attention_global(lstm_out, 75)
    # attention_mul = nonlinear_attention_relu(lstm_out, 75)
    attention_mul = nonlinear_attention_tanh(lstm_out, 75)


    attention_mul = Flatten()(attention_mul)
    a = Dropout(0.5)(attention_mul)
    a = Dense(nb_classes)(a)
    outputs = Activation('softmax')(a)
    model = Model(input=[inputs], output=outputs)

    return model




def train(max_features, nb_classes, X_train, y_train, batch_size, epochs, modelPath):

    X_train = sequence.pad_sequences(X_train, maxlen=75)
    model = model_attention_applied_after_lstm(75, max_features, nb_classes)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
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





