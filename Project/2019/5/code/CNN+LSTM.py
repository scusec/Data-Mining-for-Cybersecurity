# coding: utf-8


import numpy as np
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
import pymysql
from keras.callbacks import TensorBoard,ModelCheckpoint
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




# 连接数据库
db = pymysql.connect("127.0.0.1",
                     "root",
                     "123456",
                     "DB",
                     use_unicode=True,
                     charset="utf8"
                     )

cursor = db.cursor()
sql1 = "**************************8"
cursor.execute(sql1)
result = cursor.fetchall()




key_word = []
def get_usr():
    global key_word
    sql2 = "select usrname from nameset"
    cursor.execute(sql2)
    result_usrname=cursor.fetchall()
    for row in result_usrname:
        key_word.append(row[0])
def get_tool():
    global key_word
    sql2 = "select name from toolset"
    cursor.execute(sql2)
    result_toolname=cursor.fetchall()
    for row in result_toolname:
        key_word.append(row[0])

get_usr()
get_tool()





corpus = []
for row in result:
    corpus.append(row[1])
result = []
for sentence in corpus:
   # print(sentence.split(' '))
    text = []
    for word in sentence.split(' '):
        #print(word)
        word = word.strip()
        if word != '' and word !='The' and word in key_word:
            label = "HACK"
        
        else:
            label = "O"
        text.append([word,label])
    result.append(text) 




def addCharInformatioin(Sentences):
    for i,sentence in enumerate(Sentences):
        for j,data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0],chars,data[1]]
    return Sentences
result = addCharInformatioin(result)


trainSentences = result[:int(len(result) * .7)]
testSentences = result[int(len(result) * .7):]
#print(trainSentences)
labelSet = set()
words = {}

#每一个里面是一个三元组，分别为token，原单词，字符级表示，标签
for dataset in [trainSentences, testSentences]:
    for sentence in dataset:
        for token,char,label in sentence:
            labelSet.add(label)
            words[token.lower()] = True
# 将标签数值化
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)
    #print(label2Idx)




case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'contains_upper':7,'contains_hyphen':8,'PADDING_TOKEN':9}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')  #生成方阵，数据类型为浮点类型
#print(caseEmbeddings) #使用one-hot编码进行标签化




word2Idx = {}
wordEmbeddings = []

fEmbeddings = open("/Users/leeyn/downloads/glove.6B/glove.6B.100d.txt", encoding='utf8')
#fEmbeddings = open("word2vec.txt")
for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)
       # print(wordEmbeddings)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
        
wordEmbeddings = np.array(wordEmbeddings)

char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
    char2Idx[c] = len(char2Idx)
    
#print(char2Idx)




#wordEmbeddings




def oriSentences(sentence):
    string = ""
    for i in sentence:
        string = string+i[0]+" "
    print(string)
'''
    sentences:句子级的特征，包括每一个单词，单词对应的字符，以及该单词的标签
    word2Idx:单词级，主要就是单词在词向量中对应的key-value
    label2Idx:标签，是否是黑客才会用的词汇
    case2Idx:我们自己定义的字符的一些特征
'''
def createMatrices(sentences, word2Idx, label2Idx, case2Idx,char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']    
    oriSentences(sentences[0])   #从列表中将之前的句子再还原出来
    dataset = []
    wordCount = 0
    unknownWordCount = 0
    for sentence in sentences:
        wordIndices = []    
        caseIndices = []
        charIndices = []
        labelIndices = []
        
        for word,char,label in sentence:  
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word] #返回单词在词向量中所对应的索引
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]  
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                charIdx.append(char2Idx[x])
                
            #Get the label and map to int ,将数据都映射为整数           
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx)) #得到单词对应于什么特征
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])
        dataset.append([wordIndices, caseIndices, charIndices, labelIndices])         
    return dataset

#将句子填补
def padding(Sentences):
    maxlen = 25
    for sentence in Sentences:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen,len(x))
    for i,sentence in enumerate(Sentences):
        Sentences[i][2] = pad_sequences(Sentences[i][2],25,padding='post') #在序列的后端补齐
    return Sentences

def getCasing(word, caseLookup):   
    casing = 'other'
    numDigits = 0
    upperDigits = 0
    hyphenDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
        if char.isupper():
            upperDigits +=1
        if char == '-':
            hyphenDigits +=1
    if len(word)!=0:       
        digitFraction = numDigits / float(len(word))
        upperFraction = upperDigits/float(len(word))
    else:
        digitFraction=0
        upperFraction=0
    
    if word.isdigit():
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif hyphenDigits >0:
        casing = 'contains_hyphen' 
    elif numDigits > 0:
        casing = 'contains_digit'
    elif word.islower():
        casing = 'allLower'
    elif word.isupper():
        casing = 'allUpper'
    elif upperFraction > 0 and not word[0].isupper():
        casing = 'contains_upper'
    elif word!='' and word[0].isupper():
        casing = 'initialUpper'
    
           
    return caseLookup[casing]

train_set = padding(createMatrices(trainSentences,word2Idx, label2Idx, case2Idx,char2Idx))
test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx))

idx2Label = {v: k for k, v in label2Idx.items()}
print(idx2Label)




def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches,batch_len

train_batch,train_batch_len = createBatches(train_set)
test_batch,test_batch_len = createBatches(test_set)
print(train_batch_len)
print(test_batch_len)




words_input = Input(shape=(None,),dtype='int32',name='words_input')

words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=100,  weights=[wordEmbeddings], trainable=False)(words_input)

casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)

character_input=Input(shape=(None,25,),name='char_input')
embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)

dropout= Dropout(0.5)(embed_char_out)

conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)

maxpool_out=TimeDistributed(MaxPooling1D(25))(conv1d_out)

char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)

output = concatenate([words, casing,char])
output = Bidirectional(LSTM(20, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)

model = Model(inputs=[words_input, casing_input,character_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()




def iterate_minibatches(dataset,batch_len): 
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t,c,ch,l = dt
            l = np.expand_dims(l,-1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)
        yield np.asarray(labels),np.asarray(tokens),np.asarray(caseing),np.asarray(char)
def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0]
        pred = pred.argmax(axis=-1) #Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels
def compute_f1(predictions, correct, idx2Label): 
    label_pred = []    
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])
        
    label_correct = []    
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])
            
    
    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)
    
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1
def compute_precision(guessed_sentences, correct_sentences):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        idx = 0
        while idx < len(guessed):    
            count += 1
            if guessed[idx] == correct[idx]:
                idx += 1
                correctlyFound = True
                correctCount += 1
                while idx < len(guessed) and guessed[idx][0] == 'I': #Scan until it no longer starts with I
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False

                    idx += 1
            else:
                idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count    
    return precision
batch_num= []
ssss = 0
loss = []
prec_num = []
epochs = 30
for epoch in range(epochs):    
    print("Epoch %d/%d"%(epoch,epochs))
    a = Progbar(len(train_batch_len))
    for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
        labels, tokens, casing,char = batch    
       # model.fit([tokens, casing,char], labels, epochs=5, batch_size=32)
        loss_value = model.train_on_batch([tokens, casing,char], labels)
        loss.append(loss_value)
        ssss +=1
        batch_num.append(ssss)
        a.update(i)
        d_batch0 = random.choices(population=train_batch, k=10)
        predLabels0, correctLabels0 = tag_dataset(d_batch0)
        print(predLabels0)
        print("\n")
        print(correctLabels0)
        pre_dev0, rec_dev0, f1_dev0 = compute_f1(predLabels0, correctLabels0, idx2Label)
        prec_num.append(pre_dev0)
       
        print("\n *** Train Data: Prec: %.4f, Rec: %.4f, F1: %.4f" % (pre_dev0, rec_dev0, f1_dev0))

        
    print(' ')





