#encoding:utf-8
from tensorflow.contrib.keras.api.keras.models import *
from utils import URLDECODE
from gensim.models.word2vec import Word2Vec
from word import getVecs
import keras
import sys
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #只显示error
def init():	      
    model=load_model('bestcnn')
    w_model=Word2Vec.load("file/word2model")
    return model,w_model
def check(model,w_model,data):
    if data !=None:
        xx=[]
        filepath='file/INPUT_SHAPE'
        input_shape=[]
        with open(filepath,'r') as f :
            for line in f.readlines() :
                input_shape=int(line)
        if len(data.strip()): #判断是否是空行
            for text in URLDECODE(data) :
                #print(text)
                try:
                    xx.append(w_model[text])
                except KeyError:
                    continue
            xx=np.array(xx, dtype='float')
        if not len(xx):
            return [0]
        x=[]
        x.append(xx)
        x=np.array(x)
 
        x=keras.preprocessing.sequence.pad_sequences(x,dtype='float32',maxlen=input_shape)

        result=model.predict_classes(x, batch_size=len(x))
   
        return result
    else:
        return [0]

                

model,w_model=init()
payload = str(input("please input the payload you want to test: "))
checked=check(model,w_model,payload)
if checked[0]==1:
    print("是SQL注入的payload吧！！！")
else:
    print("不是sql注入的payload吧！！")
