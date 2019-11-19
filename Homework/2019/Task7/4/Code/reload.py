# -*- coding: utf-8 -*-
#from keras.layers import Embedding, LSTM, Dense, Activation,Dropout
from gensim.models import Word2Vec
import numpy as np
from sklearn.externals import joblib
import getData
import filter_opcode
import json

dim_num_feature = 120


def fitList(List, n):
    L = len(List)
    insert = '。'
    if L < n:
        d = n-L
        appList = [insert for i in range(d)]
        List += appList
    else:
        if L > n:
            List = List[0:n]
    #print('Fit list : ' + str(List))
    print('Fit')
    return List


def word2Vecvec(List, model):
    vec = []
    insert = [float(0) for i in range(120)]
    insert = np.asarray(insert, dtype='float32')
    for w in List:
        if w in model:
            vec.append(model[w])
        else:
            vec.append(insert)
    print('OK--word2vec')
    print('word2vec List : ' + str(vec[0]))
    return vec


def generateW2vVec(opcode_lists, n):
    w2v_model = Word2Vec.load("./120features_20minwords_10context")
    result = []
    for opcode_list in opcode_lists:
        opcode_vec = word2Vecvec(fitList(opcode_list, n), w2v_model)
        result.append(opcode_vec)
    #print('gen : ' + str(result))
    return result


# n -> length of the vector (all file should have the same length of opcodes)
def checkPath(path, model_name='LSTM_webshell_20.model', n=30):
    result_dict = {}
    raw_opcodes, success, fail = filter_opcode.trans(path)
    opcodes = getData.parse_raw_opcodes(raw_opcodes)
    opcodes_vecs = np.array(generateW2vVec(opcodes, n))
    assitant = joblib.load(model_name)
    result = assitant.predict(opcodes_vecs)
    result_dict['data'] = result
    result_dict['success'] = success
    result_dict['fail'] = fail
    return result_dict


def check(path):
    result_dict = checkPath(path)
    # print(result_dict)
    #json_obj = json.dumps(result_dict)
    # return json_obj
    return result_dict


def main():
    path = './ws/php-webshells/'
    res = check(path)
    prediction = [np.argmax(i) for i in res['data']]
    # print(prediction)
    norw = []
    for i in prediction:
        if i == 1:
            norw.append('webshell')
        else:
            norw.append('normal')
    print(norw)


if __name__ == '__main__':
    main()
