from keras.models import load_model
from keras.preprocessing import sequence
from keras.models import Sequential
import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_dict(filename):
    with open(filename, 'r') as file:
        js = file.read()
        dict = json.loads(js)
    return dict

def dga_predcit():
    model = load_model('../Model/model.h5')
    model.summary()

    valid_chars = get_dict('../Data/chars.json')
    max_dict = get_dict('../Data/max.json')

    X = [input('Please input a dga to detect: ')]
    
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=max_dict['len'])

    result = model.predict_classes(X)
    probs = model.predict_proba(X)
    print('The result of prediction: ')
    if result == [[1]]:
        print('dga_domain ' + '(' + str(probs[0][0]) + ')')
    else:
        print('good_domain ' + '(' + str(1-probs[0][0]) + ')')

if __name__ == '__main__':
    dga_predcit()