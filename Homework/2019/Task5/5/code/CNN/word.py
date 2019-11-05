#encoding:utf-8
from utils import URLDECODE
from gensim.models.word2vec import Word2Vec
import os
import numpy as np
import multiprocessing
import pickle
from keras.models import Sequential
import keras
from keras.utils import np_utils
import time,random,json
datadir="./data"

normal_data="data/train_normal.txt"
sql_data="data/train_sql.txt"
xss_data="data/train_xss.txt"
valid_normal="data/validation_normal.txt"
valid_sql="data/validation_sql.txt"
valid_xss="data/validation_xss.txt"
model_dir="file/word2model"
files="file"
num_iter=10
max_features=16

class MySentences(object): 
    def __init__(self,dirname):
        self.dirname=dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname,fname),encoding="utf-8"):
                if len(line.strip()):#判断是否是空行
                    yield URLDECODE(line) 
class getVecs(object):
    def __init__(self,filename,model,classes=None):
        self.filename=filename
        self.model=model
        self.classes=classes
        self.f_len=0
        self.max_len=0
    def __iter__(self):
        for line in open(self.filename,encoding="utf-8"):
            if len(line.strip()):#判断是否是空行
                self.f_len+=1
                xx=[]
                for text in URLDECODE(line) :
                    try:

                        xx.append(self.model[text])
                    except KeyError:
                        continue
                xx=np.array(xx, dtype='float')
                if self.max_len< len(xx):
                    self.max_len=len(xx)
                yield xx
def save_data(text,filename):
    with open(filename,'a') as f:
        for line in text:
            f.write(str(line.tolist())+"|"+str(text.classes)+"\n")
def save_len(normal,sql,xss):
    with open("./file/len",'w') as f:
            f.write(str(normal.f_len+sql.f_len+xss.f_len))
            
def maxlen(normal,sql,xss):
    max=0
    if normal.max_len<sql.max_len:
        max=sql.max_len
    else:
        max=normal.max_len
    if max<xss.max_len:
        max=xss.max_len
    return max
def predata():
    startime=time.time()
    model=Word2Vec.load(model_dir)
    x_normal=getVecs(normal_data,model,0)
    x_sql=getVecs(sql_data,model,1)
    x_xss=getVecs(xss_data,model,2)
    save_data(x_normal,"./file/x_train")
    save_data(x_sql,"./file/x_train")
    save_data(x_xss,"./file/x_train")
    save_len(x_normal,x_sql,x_xss)
    with open("./file/INPUT_SHAPE","w") as f:
        f.write(str(max_features*maxlen(x_normal,x_sql,x_xss)))
    print("save complete!")
def valid_data():
    startime=time.time()
    model=Word2Vec.load(model_dir)
    x_normal=getVecs(valid_normal,model,0)
    x_sql=getVecs(valid_sql,model,1)
    x_xss=getVecs(valid_xss,model,2)
    save_data(x_normal,"./file/x_valid")
    save_data(x_sql,"./file/x_valid")
    save_data(x_xss,"./file/x_valid")
    with open("./file/valid_len",'w') as f:
            f.write(str(x_normal.f_len+x_sql.f_len+x_xss.f_len))
def train_word2vec():
    sentences=MySentences(datadir)

    cores=multiprocessing.cpu_count()

    if os.path.exists(model_dir):
        print ("Find cache file %s" % model_dir)
        model=Word2Vec.load(model_dir)
    else:
        model=Word2Vec(size=max_features, window=5, min_count=10, iter=10, workers=cores)

        model.build_vocab(sentences)

        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.save(model_dir)
        print("save model complete!")

def read_file(filename):
    for line in open(filename):
        if len(line.strip()):
            yield line
def randan_data(filename,len):
    X=[]
    for line in open(filename):
        X.append(line)
    a=[]
    for i in range(int(len)):
        a.append(i)
    random.shuffle(a)  
    with open(filename,'w') as f:
        for i in a:
            f.write(str(X[i]))
    print("save complete!")




def data_generator(batch_size,input_shape,filename):
    while True:
        cnt=0
        X=[]
        Y=[]
        for line in open(filename):
            [x,y]=line.split("|")
            x=json.loads(x)
            y=json.loads(y)
            X.append(x)
            Y.append(y)
            cnt+=1
            if cnt==batch_size:
                cnt=0
                X=np.array(X)
                Y=np.array(Y)
                X=keras.preprocessing.sequence.pad_sequences(X,
                    maxlen=input_shape, dtype='float32')
                
                Y=np_utils.to_categorical(Y,3)

                yield (X,Y)
                X=[]
                Y=[]
def batch_generator(next_batch,data_size):
    while True:
            X,Y=next(next_batch)
            yield (X,Y)
            
if __name__=="__main__":
    #save_data(files)
    train_word2vec()
    predata()
    for i in open("./file/len"):
        randan_data("./file/x_train",i)
    valid_data()
    for i in open("./file/valid_len"):
        randan_data("./file/x_valid",i)