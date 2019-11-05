import sklearn.ensemble as ek
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import numpy as np
import pyprind
import re
from collections import Counter
import math
import traceback
model_dir = r'C:\Users\86151\Desktop\Data Mining  Homework\Wee 8\ML-for-SQL-Injection\ML_for_SQL\data\models\model.model'


def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())


def getFeatures(url,label):
    result = []
    url = str(url)
    result.append(url)
    num_len=0
    capital_len=0
    key_num=0
    feature3=0
    num_len=len(re.compile(r'\d').findall(url))
    try:
        if len(url)!=0:
            num_f=num_len/len(url)#数字字符频率
        capital_len=len(re.compile(r'[A-Z]').findall(url))
        if len(url)!=0:
            capital_f=capital_len/len(url)#大写字母频率
        url=url.lower()
        key_num=url.count('and%20')+url.count('or%20')+url.count('xor%20')+url.count('sysobjects%20')+url.count('version%20')+url.count('substr%20')+url.count('len%20')+url.count('substring%20')+url.count('exists%20')
        key_num=key_num+url.count('mid%20')+url.count('asc%20')+url.count('inner join%20')+url.count('xp_cmdshell%20')+url.count('version%20')+url.count('exec%20')+url.count('having%20')+url.count('unnion%20')+url.count('order%20')+url.count('information schema')
        key_num=key_num+url.count('load_file%20')+url.count('load data infile%20')+url.count('into outfile%20')+url.count('into dumpfile%20')
        if len(url)!=0:
            space_f=(url.count(" ")+url.count("%20"))/len(url)#空格百分比
            special_f=(url.count("{")*2+url.count('28%')*2+url.count('NULL')+url.count('[')+url.count('=')+url.count('?'))/len(url)
            prefix_f=(url.count('\\x')+url.count('&')+url.count('\\u')+url.count('%'))/len(url)
        result.append(len(url))
        result.append(key_num)
        result.append(capital_f)
        result.append(num_f)
        result.append(space_f)
        result.append(special_f)
        result.append(prefix_f)
        result.append(entropy(url))
        result.append(str(label))
        return result
    except:
        traceback.print_exc()
        exit(-1)


def plot_feature_importances(feature_importances,title,feature_names):
#     将重要性值标准化
    feature_importances = 100.0*(feature_importances/max(feature_importances))
#     将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))
#     让X坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0])+0.5

    # plt.figure(figsize=(16,4))
    # plt.bar(pos,feature_importances[index_sorted],align='center')
    # plt.xticks(pos,feature_names[index_sorted])
    # plt.ylabel('Relative Importance')
    # plt.title(title)
    # plt.show()


if __name__ == '__main__':
    # 提取特征
    df = pd.read_csv("data/dataset.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    featureset = []
    featureSet = pd.DataFrame(columns=('url','length','key_num','capital_f','num_f','space_f','special_f','prefix_f',
                                       'entropy','label'))
    pbar = pyprind.ProgBar(len(df),title='特征提取进度')
    try:
        for i in range(0, len(df)):
            features = getFeatures(df['url'].loc[i],df['label'].loc[i])
            featureSet.loc[i] = features
            pbar.update()
    except:
        traceback.print_exc()
        exit(-1)
    print(featureSet.head(5))
    print(featureSet.groupby(featureSet['label']).size())

    # 开始训练
    X = featureSet.drop(['url','label'],axis=1).values
    y = featureSet['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)

    model = { "DecisionTree":tree.DecisionTreeClassifier(max_depth=10),
             "RandomForest":ek.RandomForestClassifier(n_estimators=50),
             "Adaboost":ek.AdaBoostClassifier(n_estimators=50),
             "GradientBoosting":ek.GradientBoostingClassifier(n_estimators=50),
             "GNB":GaussianNB(),
             "LogisticRegression":LogisticRegression()
    }

    results = {}
    for algo in model:
        clf = model[algo]
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        print ("%s : %s " %(algo, score))
        results[algo] = score
    winner = max(results, key=results.get)
    print('winner: '+winner)

    clf = model[winner]

    res = clf.predict(X)
    mt = confusion_matrix(y, res)
    FP=(mt[0][1] / float(sum(mt[0])))*100
    FN= (mt[1][0] / float(sum(mt[1]))*100)
    TP=(mt[0][0] / float(sum(mt[0])))*100
    TN=(mt[1][1] / float(sum(mt[1]))*100)
    print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
    print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))
    print("True positive rate : %f %%" % ((mt[0][0] / float(sum(mt[0])))*100))
    print('True negative rate : %f %%' % ( (mt[1][1] / float(sum(mt[1]))*100)))
    print('Precision rate: %f ' % (TP/(TP+FP)))
    print('Recall rate: %f ' % (TP/(TP+FN)))

    with open(model_dir,"wb+") as f:
            pickle.dump(clf,f,protocol=2)
    print("wirte to ",model_dir)



    # 打印特征重要性
    feature_names = ['length','key_num','capital_f','num_f','space_f','special_f','prefix_f','entropy']
    feature_names = np.array(feature_names)
    print(clf.feature_importances_)
    plot_feature_importances(clf.feature_importances_,winner,feature_names)
