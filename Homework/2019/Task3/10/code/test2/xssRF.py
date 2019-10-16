# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn import metrics
import joblib
import operator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
#随机森林
from sklearn.ensemble import RandomForestClassifier
# 决策树
from sklearn.tree import DecisionTreeClassifier
# 极限随机树
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import preprocessing

# 产生样本数据集
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


x = []
y = []

def get_len(url):#获取长度
    return len(url)

def get_url_count(url):
    if re.search('(https://)|(https://)',url,re.IGNORECASE):
        return 1
    else:
        return 0
    
    
def get_evil_char(url):
    return len(re.findall("[<>,\'\"/]", url, re.IGNORECASE))


def get_evil_word(url):
    return len(re.findall("(alert)|(script=)(%3c)|(%3e)|(%20)|(onerror)|(onload)|(eval)|(src=)|(prompt)", url, re.IGNORECASE))


def get_last_char(url):
    if re.search('/$', url, re.IGNORECASE) :
        return 1
    else:
        return 0
    
    
def get_feature(url):
    return [get_len(url),get_url_count(url),get_evil_char(url),get_evil_word(url),get_last_char(url)]


def do_metrics(y_test,y_pred):#打印结果
    print ("准确率:%s",metrics.accuracy_score(y_test, y_pred))
    print ("召回率:%s",metrics.recall_score(y_test, y_pred))

 
def etl(filename, data, isxss):
    with open(filename,encoding='gb18030',errors='ignore') as f:
        for line in f:
            f1 = get_len(line)
            f2 = get_url_count(line)
            f3 = get_evil_char(line)
            f4 = get_evil_word(line)
            data.append([f1, f2, f3, f4])
            if isxss:
                 y.append(1)
            else:
                y.append(0)
    return data
etl('xssed.csv', x, 1)#加载正常数据并打标1到data（也就是x）中
etl('dmzo_nomal.csv', x, 0)#加载异常数据并打标0到data（也就是x）中


#划分数据集 30%测试
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=0)

"""print("线性SVM:\n");
clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)#线性SVM
y_pred = clf.predict(x_test)
do_metrics(y_test, y_pred)

#标准化缩放
min_max_scaler = preprocessing.MinMaxScaler()  #标准化缩放
#x_min_max=min_max_scaler.fit_transform(x)
X_train_minmax = min_max_scaler.fit_transform(x_train)
X_test_minmax = min_max_scaler.transform(x_test)

clf = svm.SVC(kernel='linear', C=1).fit(X_train_minmax, y_train)#线性SVM
y_pred_minmax = clf.predict(X_test_minmax)
print("最大最小标准化后：")
do_metrics(y_test, y_pred_minmax)        

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pre = knn.predict(x_test)
#score1=knn.score(x_test,y_test,sample_weight=None)
print("knn:\n")
do_metrics(y_test, y_pre)        


rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
predict_results=rfc.predict(x_test)
print("随即森林:\n")
do_metrics(y_test, predict_results)
"""

       # ==================决策树、随机森林、极限森林对比===============



clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
clf.fit(x_train,y_train)
y_pre=clf.predict(x_test)
print('决策树准确率：')
do_metrics(y_test, y_pre)


clf = RandomForestClassifier(n_estimators=10,max_features=2)
clf.fit(x_train, y_train)
y_pre=clf.predict(x_test)
print('随机森林准确率：')
do_metrics(y_test, y_pre)


clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2,random_state=0)
clf.fit(x_train,y_train)
y_pre=clf.predict(x_test)
print('极限随机树准确率：')
do_metrics(y_test, y_pre)
print('模型中各属性的重要程度：',clf.feature_importances_)  


#使用网格搜索确定要建立的基学习器个数
clf = GridSearchCV(RandomForestClassifier(max_features=None),
                   param_grid=({"n_estimators":range(1,101,10)}),cv=10)
clf.fit(x_train,y_train)
#print(clf.best_params_)
#再使用网格搜索来确定决策树的参数
clf2 = GridSearchCV(RandomForestClassifier(n_estimators=81),
                    param_grid=({"max_depth":range(1,10)}))
clf2.fit(x_train,y_train)
#print(clf2.best_params_)
#根据最大层数8，最多棵树81，建立最终的随机森林来预测
rf = RandomForestClassifier(n_estimators=81,max_depth=8,max_features=None)
rf.fit(x_train,y_train)
y_hat = rf.predict(x_test)
print()
do_metrics(y_test, y_hat)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
