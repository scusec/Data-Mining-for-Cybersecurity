from gensim.models.word2vec import Word2Vec
from utils import segment
import pickle
import numpy as np

FAST_LOAD = False
embedding_size =  100  # 隐层的维度
vec_dir = "bins/word2vec.model"  # word2vec存放位置
window = 5  # 上下文距离
iter_num = 30  # word2vec的迭代数
min_num = 1  # word2vec的最少出现次数
max_voc =  1000  # 最大字典数
time_step = 30  # 时序，即单句的最大seg长

lens_dir = 'bins/lens.pkl'
y_train_dir = 'bins/y_train.npy'
x_train_dir = 'bins/x_train.npy'
classes_voc_dir = 'bins/classes_voc.pkl'
classes = {}  # 储存类到index的映射


def most_similar(w2v_model, word, topn=10):
    try:
        similar_words = w2v_model.wv.most_similar(word, topn=topn)
    except:
        print(word, "not found in Word2Vec model!")
    return similar_words

if __name__=='__main__':
    if (not FAST_LOAD):
        y = []
        with open('data/names.txt') as f:
            names = f.readlines()
            for i in range(len(names)):
                names[i] = names[i].strip()
                classes[names[i]] = i

        with open(classes_voc_dir, 'wb') as f:
            pickle.dump(classes,f)

        payloads = []
        payloads_seged = []
        lens = []
        for name in names:
            print(name)
            with open('data/'+name+'.txt', encoding='utf8') as fread:
                while(1):
                    payload = fread.readline()
                    if(payload=='\r\n' or payload=='\n' or payload=='\r'):
                        continue
                    if(not payload):
                        break
                    payload=payload.strip()
                    
                    payloads.append(payload)
                    y.append(classes[name])
                   

        y = np.array(y)
        np.save(y_train_dir, y)

        for payload in payloads:
            tempseg = segment(payload)
            if(tempseg==[]):
                print(payload)
            payloads_seged.append(tempseg)
            lens.append(len(tempseg))

    
        with open(lens_dir, 'wb') as f:
            pickle.dump(lens,f)

        model = Word2Vec(
            payloads_seged,
            size=embedding_size,
            iter=iter_num, sg=1,
            min_count=min_num,
            max_vocab_size=max_voc
        )

        model.save(vec_dir)

        x = []
        tempvx = []
        for payload in payloads_seged:
            for word in payload:
                try:
                    tempvx.append(model.wv.get_vector(word))
                except KeyError as e:
                    tempvx.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位
            tempvx=np.array(tempvx)
            if(tempvx.shape[0]==0):
                print(payload)
            x.append(tempvx)
            # print(tempvx.shape)
            tempvx=[]
        

        # 字符串向量长度填充
        lenth=time_step
        for i in range(y.shape[0]):
            if (x[i].shape[0] < lenth):
                try:
                    x[i]=np.pad(x[i], ((0, lenth - x[i].shape[0]),
                                    (0, 0)), 'constant', constant_values=0)
                except ValueError as e:
                    print(i)
                    print(x[i].shape)
                    print(x[i])
                    exit()
            elif (x[i].shape[0] > lenth):
                x[i]=x[i][0:lenth]
        x=np.array(list(x))
        # print(x.shape)
        np.save(x_train_dir, x)

    else:
        model=Word2Vec.load(vec_dir)

    # print(model.wv.vocab)
    print(most_similar(model,'alert('))
    # print(most_similar(model,'select'))

