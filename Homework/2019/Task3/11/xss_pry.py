
#coding: utf-8

import re
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn import metrics

x = [] #特征值矩阵
y = [] #样本标签

### 特征统计
def get_len(url):
    return len(url)

def isURL(param):
    if re.search('(http://)|(https://)',param,re.IGNORECASE):#正则表达匹配
        return 1
    else:
        return 0

def countChar(param):
    return len(re.findall("[<>()\'\"/]",param,re.IGNORECASE))#正则表达匹配

def countWord(param):
    return len(re.findall('(alert)|(scripts=)(%3ac)|(%3e)|(%20)|(onerror)|(onload)|(eval)|(src=)|(prompt)|(iframe)|(java)',param,re.IGNORECASE))#正则表达匹配

### 向量化
def getMatrix(filename, data, isxss):
    with open(filename) as fd:
        for line in fd:
            x1 = get_len(line)
            x2 = isURL(line)
            x3 = countChar(line)
            x4 = countWord(line)
            data.append([x1,x2,x3,x4])
            if isxss:
                y.append(1)
            else:
                y.append(0)

getMatrix('/Users/dqy/XSS/xssed.csv',x,1)
getMatrix('/Users/dqy/XSS/dmzo_normal.csv',x,0)

### 训练
#### 拆分数据
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
clf = svm.SVC(kernel='linear',C=1).fit(x_train,y_train)
#### SVM训练
y_pred = clf.predict(x_test)

print ("metrics.accuracy_score:")
print (metrics.accuracy_score(y_test,y_pred))
print ("metrics.recall_score:")
print (metrics.recall_score(y_test,y_pred))

### 测试
line = input("test: ");
x1 = get_len(line)
x2 = isURL(line)
x3 = countChar(line)
x4 = countWord(line)
test_x=[[x1,x2,x3,x4]]
#test_y.append(1)
if(clf.predict(test_x)=="0"):
    print("Benign")
else:
    print("Malicious XSS")


