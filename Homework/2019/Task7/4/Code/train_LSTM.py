# -*- coding: utf-8 -*-
# 句向量训练者称为出题者assitant，标签训练者称为student
import getData
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from gensim.models import Word2Vec
import numpy as np
from keras import metrics


dim_num_feature = 120

# 用于限制长度


def magicFit(List, n):
    L = len(List)
    # 额外添加
    insert = ' '
    if L < n:
        d = n-L
        appList = [insert for i in range(d)]
        List += appList
    else:
        if L > n:
            List = List[0:n]
    return List


def sen2Vec(List, model):
    vec = []
    insert = [float(0) for i in range(120)]
    insert = np.asarray(insert, dtype='float32')
    for w in List:
        if w in model:
            vec.append(model[w])
        else:
            vec.append(insert)
    return vec


w2v_model = Word2Vec.load("./data/120features_20minwords_10context")
train_comments, train_labels = getData.getAllData(
    "./data/webshell.json", "./data/normal.json")


count_list = []
commentVec_list = []
n = 30
j = 0

for comment in train_comments:
    if(j % 100 == 0):
        print("%d of %d total" % (j, len(train_comments)))
    count_list.append(len(comment))
    comment_vec = sen2Vec(magicFit(comment, n), w2v_model)
    commentVec_list.append(comment_vec)
    j += 1


commentVec_list = np.array(commentVec_list)
train_labels = np.array(train_labels)

indices = np.arange(commentVec_list.shape[0])

np.random.shuffle(indices)
data = commentVec_list[indices]
labels = train_labels[indices]


VALID_SAMPLE_SPLIT = 0.1
nb_vali_samples = int(VALID_SAMPLE_SPLIT * data.shape[0])
x_train = data[:-nb_vali_samples]
y_train = labels[:-nb_vali_samples]
print(x_train.shape[0], x_train.shape[1], x_train.shape[2])
y_train = getData.getOneHot(y_train)

x_val = data[-nb_vali_samples:]
y_val = labels[-nb_vali_samples:]
y_val = getData.getOneHot(y_val)

'''-----------Train Model-------------'''
assistant = Sequential()
assistant.add(LSTM(128, activation='tanh', input_shape=(
    x_train.shape[1], x_train.shape[2])))
assistant.add(Dropout(0.2))
assistant.add(Dense(2, activation='softmax'))
assistant.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', "mae"])
assistant.summary()

assistant.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=12, batch_size=128)

#pickle_file = open('LSTM_webshell_20.model.pik','wb')
# pickle.dump(assistant,pickle_file)
# pickle_file.close()
