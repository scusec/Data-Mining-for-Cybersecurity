import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
import joblib

feature = pd.read_csv("./Data/all_data.csv")
arr=feature.values
data = np.delete(arr, -1, axis=1) #删除最后一列
#print(arr)
target=arr[:,6]
#随机划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.3,random_state=3)
#模型
clf=tree.DecisionTreeClassifier(criterion="entropy",max_depth=1)
clf.fit(X_train,y_train)#训练模型
joblib.dump(clf, './Model/DecisionTree.model')
print("DecisionTree.model 被保存到：'Model/DecisionTree.model'目录下")
#clf = joblib.load('svm.model')
y_pred=clf.predict(X_test)#预测
print("Decison Tree:")
print("y_pred:%s"%y_pred)
print("y_test:%s"%y_test)
#Verify
print('Precision:%.3f' %metrics.precision_score(y_true=y_test,y_pred=y_pred))#查全率
print('Recall:%.3f' %metrics.recall_score(y_true=y_test,y_pred=y_pred))#查准率