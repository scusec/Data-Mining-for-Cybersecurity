import string
import re
import codecs
import jieba
import numpy as np
from tensorflow.contrib import learn
import tensorflow as tf
import collections
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

datapath=r".\app\sentiment\\"

def delstopword(line,stopkey):
    wordList = line.split(' ')          
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopkey:
            if word != '\t':
                sentence += word + " "
    return sentence.strip()

'''
预测函数
参数：文本字符串
返回：预测类别对应的概率（如：[[6.2826526e-04 7.2620180e-03 7.5219780e-01]]），预测的类别（[0]积极、[1]中性、[2]消极）
'''
def sent_predict(str):
    stopkey = [w.strip() for w in codecs.open(datapath+r'stopWord.txt', 'r', encoding='utf-8').readlines()]
    text = delstopword(" ".join(i for i in jieba.lcut(str.strip(),cut_all=False)), stopkey)
  
    max_document_length=150
    vocab = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab.fit_transform([text])
    pred_dict = collections.OrderedDict(vocab.vocabulary_._mapping)

    keys = []
    values = []
    with codecs.open(datapath+r"vocabulary.txt","r","utf-8") as f:
        for line in f:
            temp = line.split(',')
            keys.append(temp[0])
            values.append(temp[1].replace('\n',''))
    vocab_dict = dict(zip(keys, values))  

    x = [0]*max_document_length

    for key, value in pred_dict.items():
        for key1, value1 in vocab_dict.items():
            if (key==key1):
                x[int(value)-1] = int(value1)
                break
    
    x = np.array([x])

    model = load_model(datapath+r'sentiment_analysis_lstm.h5')
    predictions=model.predict(x)
    predict_class = model.predict_classes(x)
    return predictions[0][0], predictions[0][1], predictions[0][2], predict_class[0]


if __name__ == '__main__':
    positive_prob, confidence, negative_prob, sentiment = sent_predict("这门课终于结束了，有点不舍，又有点开心。")
    