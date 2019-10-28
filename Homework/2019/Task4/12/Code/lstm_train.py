import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import pickle
import json
import os
import data_preprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_model(max_features, max_len):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_len))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    return model

def run(max_epoch=1, nfolds=10, batch_size=128):
    if os.path.isfile('../Data/data.csv') and os.path.isfile('../Data/max.json'):
        df = pd.read_csv('../Data/data.csv', engine='python', header=None)
        df.dropna(axis=0, how='any')
        X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values
        with open('../Data/max.json', 'r') as file:
            js = file.read()
            max_dict = json.loads(js)
            max_features = max_dict['feature']
            max_len = max_dict['len']
    else:
        X, y, max_features, max_len = data_preprocess.preprocess()

    labels = ['bad' if x == 0 else 'good' for x in y]

    final_data = []

    for fold in range(nfolds):
        print("fold %u/%u" % (fold + 1, nfolds))
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels,
                                                                           test_size=0.2)

        print('Build model...')
        model = build_model(max_features, max_len)

        print("Train...")
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1)

            t_probs = model.predict_proba(X_holdout)
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

            print('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict_proba(X_test)

                out_data = {'y': y_test, 'labels': label_test, 'probs': probs, 'epochs': ep,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
            else:
                if (ep - best_iter) > 2:
                    break
        
        model.save('../Model/model.h5')
        model.save_weights('../Model/model_weights.h5')
        
        final_data.append(out_data)
    pickle.dump(final_data, open('../Model/results.pkl', 'wb'))

    return final_data

if __name__ == '__main__':
    run(nfolds=1)
