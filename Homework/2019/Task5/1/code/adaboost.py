# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:06:57 2017

@author: wf
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from featurepossess import generate
from sklearn.externals import joblib

sql_matrix=generate("./data/sqlnew.csv","./data/sql_matrix.csv",1)
nor_matrix=generate("./data/normal_less.csv","./data/nor_matrix.csv",0)

df = pd.read_csv(sql_matrix)
df.to_csv("./data/all_matrix.csv",encoding="utf_8_sig",index=False)
df = pd.read_csv( nor_matrix)
df.to_csv("./data/all_matrix.csv",encoding="utf_8_sig",index=False, header=False, mode='a+')

feature_max = pd.read_csv('./data/all_matrix.csv')
arr=feature_max.values
data = np.delete(arr, -1, axis=1) #删除最后一列
#print(arr)
target=arr[:,7]
#随机划分训练集和测试集
train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.3,random_state=3)
#模型
model1=DecisionTreeClassifier(max_depth=5)
model2=GradientBoostingClassifier(n_estimators=100)
model3=AdaBoostClassifier(model1,n_estimators=100)
model1.fit(train_data,train_target)#训练模型
model2.fit(train_data,train_target)#训练模型
model3.fit(train_data,train_target)#训练模型
joblib.dump(model2, './file/GBDT.model')#梯度提升书算法
print("GBDT.model has been saved to 'file/GBDT.model'")

joblib.dump(model3, './file/Adaboost.model')
print("Adaboost.model has been saved to 'file/Adaboost.model'")
#clf = joblib.load('svm.model')
y_pred1=model2.predict(test_data)#预测
print("y_pred:%s"%y_pred1)
print("test_target:%s"%test_target)
#Verify
print("GBDT:")
print('Precision:%.3f' %metrics.precision_score(y_true=test_target,y_pred=y_pred1))#查全率
print('Recall:%.3f' %metrics.recall_score(y_true=test_target,y_pred=y_pred1))#查准率
print(metrics.confusion_matrix(y_true=test_target,y_pred=y_pred1))#混淆矩阵

y_pred2=model3.predict(test_data)#预测
print("y_pred:%s"%y_pred2)
print("test_target:%s"%test_target)
#Verify
print("Adaboost:")
print('Precision:%.3f' %metrics.precision_score(y_true=test_target,y_pred=y_pred2))#查全率
print('Recall:%.3f' %metrics.recall_score(y_true=test_target,y_pred=y_pred2))#查准率
print(metrics.confusion_matrix(y_true=test_target,y_pred=y_pred2))#混淆矩阵


