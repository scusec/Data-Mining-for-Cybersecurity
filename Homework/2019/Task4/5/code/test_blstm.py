from tensorflow.contrib.keras.api.keras.models import *
from keras.preprocessing.sequence import pad_sequences
import pickle

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100

model = load_model("model-blstm.h5")

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

domain=str(input("请输入你想要测试的域名:"))
data =[]
data.append(domain)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(data)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

predict_test=model.predict(data)
for i in predict_test:
    if i[0]<0.5:
        print("不是DGA吧，概率只有："+str(i[0]))
    else:
        print("应该是DGA了，是DGA的概率为: "+str(i[0]))