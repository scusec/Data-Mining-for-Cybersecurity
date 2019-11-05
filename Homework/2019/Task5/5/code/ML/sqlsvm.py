# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:00:50 2017

@author: wf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from featurepossess import generate
from sklearn.externals import joblib

sql_matrix=generate("./data/sqlnew.csv","./data/sql_matrix.csv",1)
nor_matrix=generate("./data/normal_less.csv","./data/nor_matrix.csv",0)

df = pd.read_csv(sql_matrix)
df.to_csv("./data/all_matrix.csv",encoding="utf_8_sig",index=False)
df = pd.read_csv( nor_matrix)
df.to_csv("./data/all_matrix.csv",encoding="utf_8_sig",index=False, header=False, mode='a+')

# with open('sql_matrix', 'ab') as f:
#     f.write(open('nor_matrix', 'rb').read())
feature_max = pd.read_csv('./data/all_matrix.csv')
arr=feature_max.values
data = np.delete(arr, -1, axis=1) #删除最后一列
#print(arr)
target=arr[:,7]
#随机划分训练集和测试集
train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.3,random_state=8)
clf = SVC(kernel='rbf')#创建分类器对象，采用概率估计，默认为False
clf.fit(train_data, train_target)#用训练数据拟合分类器模型
joblib.dump(clf, './file/svm.model')
print("svm.model has been saved to 'file/svm.model'")
#clf = joblib.load('svm.model')
y_pred=clf.predict(test_data)#预测
print("y_pred:%s"%y_pred)
print("test_target:%s"%test_target)
#Verify
print('Precision:%.3f' %metrics.precision_score(y_true=test_target,y_pred=y_pred))#查全率
print('Recall:%.3f' %metrics.recall_score(y_true=test_target,y_pred=y_pred))#查准率
print(metrics.confusion_matrix(y_true=test_target,y_pred=y_pred))#混淆矩阵
#print('F1:%.3f' %metrics.f1_score(y_true=test_target,y_pred=y_pred))#F1度量
#fpr,tpr,thresholds=metrics.roc_curve(y_true=test_target,y_score=y_pred)
#print(fpr,tpr,thresholds)
#print('auc:%.3f' %metrics.auc(fpr,tpr))
#print('auc:%.3f' %metrics.roc_auc_score(y_true=test_target,y_score=y_pred))
#plt.figure(1)
#plt.axis([0,1,0,1])#设置横轴纵轴最大坐标
#plt.plot([0,1],[0,1],'k--')#绘制对角线曲线
#plt.plot(fpr,tpr,label='ROCcurve')#有问题，只有3个点
#plt.xlabel('False positive rate')#x轴标签
#plt.ylabel('True positive rate')#y轴标签
#plt.title('ROC curve')
#plt.legend(loc='best')#生成图例
#plt.show()#显示图形
