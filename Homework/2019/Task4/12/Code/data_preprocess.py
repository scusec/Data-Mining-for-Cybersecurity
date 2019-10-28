from keras.preprocessing import sequence
import pickle
import numpy as np
import json
import pandas as pd

def dict2json(dict, filename):
    js = json.dumps(dict)
    with open(filename, 'w') as file:
        file.write(js)

def preprocess():
    data = pickle.load(open('../Data/traindata.pkl', 'rb'))

    X = []
    y = []
    for x in data:
        X.append(x[1])
        y.append(x[0])

    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X)))}
    print(valid_chars)
    dict2json(valid_chars, '../Data/chars.json')

    max_features = len(valid_chars) + 1
    max_len = np.max([len(x) for x in X])
    max_dict = {'features': int(max_features), 'len': int(max_len)}
    print(max_dict)
    dict2json(max_dict, '../Data/max.json')

    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=max_len)
    y = np.array(y)

    dataset = np.column_stack((X, y))    
    pd_data = pd.DataFrame(dataset)
    print(pd_data)
    pd_data.to_csv('../Data/data.csv')
    
    return X, y, max_features, max_len

if __name__ == '__main__':
    preprocess()
