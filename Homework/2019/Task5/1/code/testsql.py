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

def  test_c(flag,sql_flag):
    sql_dir = "./data/sql_test.csv"
    nor_dir = "./data/normal_test.csv"
    allm_dir = "./data/alltest_matrix.csv"
    if flag=='1' and sql_flag=='0':
        nor_matrix = generate(nor_dir, "./data/nor_matrix.csv", 0)
        return nor_matrix
    elif flag=='1' and sql_flag=='1':
        sql_matrix = generate(sql_dir, "./data/sqltest_matrix.csv", 1)
        return sql_matrix
    else:
        sql_matrix=generate(sql_dir,"./data/sqltest_matrix.csv",1)
        nor_matrix=generate(nor_dir,"./data/nortest_matrix.csv",0)
        df = pd.read_csv(sql_matrix)
        df.to_csv(allm_dir,encoding="utf_8_sig",index=False)
        df = pd.read_csv( nor_matrix)
        df.to_csv(allm_dir,encoding="utf_8_sig",index=False, header=False, mode='a+')
        return allm_dir
def test_data(allm_dir):
    feature_max = pd.read_csv(allm_dir)
    arr=feature_max.values
    test_data = np.delete(arr, -1, axis=1) #删除最后一列
    #print(arr)
    test_target=arr[:,7]
    return test_data,test_target

if __name__=="__main__":
    while(1):
        model_name=input("请输入要选择的模型名称：")
        clf = joblib.load('./file/'+model_name)
        print(model_name," has been loaded")
        flag=input("请输入测试文件个数：")
        sql_flag=input("请输入样本类型：")
        mode=test_c(flag,sql_flag)
        test_data,test_target=test_data(mode)
        y_pred=clf.predict(test_data)#预测
        print("y_pred:%s"%y_pred)
        print("test_target:%s"%test_target)
        #Verify
        print('Precision:%.3f' %metrics.precision_score(y_true=test_target,y_pred=y_pred))#查全率
        print('Recall:%.3f' %metrics.recall_score(y_true=test_target,y_pred=y_pred))#查准率
        print(metrics.confusion_matrix(y_true=test_target,y_pred=y_pred))#混淆矩阵

