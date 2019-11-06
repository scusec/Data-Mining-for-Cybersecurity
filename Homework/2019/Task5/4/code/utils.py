from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

import nltk
import re
from urllib.parse import unquote


def makeonehot(dictionary):
    values = array(dictionary)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)

    joblib.dump(label_encoder, 'labelenc.pkl')
    joblib.dump(onehot_encoder, 'onehotenc.pkl')


def word2vec(word):
    word = list(word)
    label_encoder = joblib.load('labelenc.pkl')
    onehot_encoder = joblib.load('onehotenc.pkl')
    integer_encoded = label_encoder.transform(word)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.transform(integer_encoded)
    return onehot_encoded


def load_bad(path):
    data = np.loadtxt(path, dtype=str, delimiter='	')
    return data


def load_good(path):
    data = np.loadtxt(path, dtype=str, delimiter=',')
    return data[:, 1]


def shuffle_data(train_data, train_target):
    batch_size = len(train_target)
    index = [i for i in range(0, batch_size)]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0, batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target


def load_lines(path):
    data = np.loadtxt(path, dtype=str, delimiter='\n')
    return data


def evaluate(y, y_pred, name='test'):
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    f1 = f1_score(y, y_pred, average='binary')
    print('name:', name)
    print('Accuracy: ', acc)
    print('Precision: ', prec)
    print('Recall: ', recall)
    print('F1-score: ', f1)
    with open('result/result.txt', 'a') as f:
        print('name:', name, file=f)
        print('Accuracy: ', acc, file=f)
        print('Precision: ', prec, file=f)
        print('Recall: ', recall, file=f)
        print('F1-score: ', f1, file=f)
        print('********************************\n', file=f)


def get_domain(url):
    pos1 = url.find('.')
    return url[:pos1]  # 提取域名


def segment(payload,debug=False):  # 分词函数
        # 数字泛化为"0"
    payload = payload.lower()
    payload = unquote(unquote(payload))
    if(debug):
        print(payload)
    payload, _ = re.subn(r'\d+', "0", payload)
    payload, _ = re.subn(
        r'{[\w./]*}', "{file}", payload)
    payload, _ = re.subn(
        r'0x[0-9a-fA-F]+', "0x0", payload)
    # 替换url为”http://u
    payload, _ = re.subn(
        r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)

    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
        |'
        |"
        |#
        |{[\w./]*}
        |=
        |[0-9]
    '''
    segs=nltk.regexp_tokenize(payload, r)
    if(segs==[]):
        segs=['#']
    return segs


def eva_with_mat(conf_mat, names):

    prec_macro = 0
    recall_macro = 0
    f1_macro = 0
    tp_all = 0
    fn_all = 0
    fp_all = 0
    for i in range(conf_mat.shape[0]):
        tp_all += conf_mat[i, i]
        fn_all += np.sum(conf_mat[i])-conf_mat[i, i]
        fp_all += np.sum(conf_mat[:, i])-conf_mat[i, i]
        if(np.sum(conf_mat[:, i]) != 0):
            precision = conf_mat[i, i]/np.sum(conf_mat[:, i])
        else:
            precision = 0
        if(np.sum(conf_mat[i]) != 0):
            recall = conf_mat[i, i]/np.sum(conf_mat[i])
        else:
            recall = 0
        if(precision+recall == 0):
            f1 = 0
        else:
            f1 = 2*precision*recall/(precision+recall)

        prec_macro += precision
        recall_macro += recall
        f1_macro += f1

        print(names[i])
        print('precision:', precision)
        print('recall:', recall)
        print('f1-score:', f1)
        print('**************************************')

    # prec_macro = prec_macro/conf_mat.shape[0]
    # recall_macro = recall_macro/conf_mat.shape[0]
    # f1_macro = f1_macro/conf_mat.shape[0]
    # print('macro')
    # print('precision:', prec_macro)
    # print('recall:', recall_macro)
    # print('f1-score:', f1_macro)
    # print('**************************************')
    #
    # prec_micro = tp_all/(tp_all+fp_all)
    # recall_micro = tp_all/(tp_all+fn_all)
    # f1_micro = 2*prec_micro*recall_micro/(prec_micro+recall_micro)
    # print('micro')
    # print('precision:', prec_micro)
    # print('recall:', recall_micro)
    # print('f1-score:', f1_micro)
    # print('**************************************')
