import pandas as pd
import numpy as np
from random import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import joblib
import csv


#加载数据集
def loadDataSet():
    permissionsList = []
    classVec = []

    file = open("./data/malicious_permissions.csv", "r", newline="")
    readers = csv.reader(file)
    for row in readers:
        permissionsList.append(row)
    file2 = open("./data/normal_permissions.csv", "r", newline="")
    readers2 = csv.reader(file2)
    for row in readers2:
        permissionsList.append(row)

    shuffle(permissionsList)
    for permissions in permissionsList:
        classVec.append(int(permissions[0]))
        del permissions[0]

    file.close()
    file2.close()
    return permissionsList, classVec


#建立词汇表
def createVocabList(permissionsSet):
     vocablaryList = set()
     for permissions in permissionsList:
         vocablaryList = vocablaryList | set(permissions)
     return list(vocablaryList)


#特征转化为词向量
def setOfWords2Vec(vocablarySet, inputSet):
    returnVec = [0] * len(vocablarySet)
    for word in inputSet:
        if word in vocablarySet:
            returnVec[vocablarySet.index(word)] = 1
        else:
            continue
    return returnVec


if __name__ == "__main__":
    permissionsList, classVec = loadDataSet()
    for i in range(5):
        print(classVec[i], end="\t")
        print(permissionsList[i])

    vocablaryList = createVocabList(permissionsList)
    file = open("vacablaryList.txt", "w", newline="")
    for v in vocablaryList:
        file.write(v + "\n")
    file.close()

    Matrix = []
    for per in permissionsList:
        Matrix.append(setOfWords2Vec(vocablaryList, per))

    X_train, X_test, y_train, y_test = train_test_split(Matrix, classVec, test_size=0.2, random_state=0)
    clf = MultinomialNB(alpha=1.0)
    print("开始训练…………")
    clf.fit(X_train, y_train)
    print("训练结束！")
    y_pre = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pre)
    recall = recall_score(y_test, y_pre, average="weighted")
    joblib.dump(clf, './Model/NB.pkl')
    print("模型保存成功！")

    print("Accuracy: %.5f" % accuracy)
    print("Recall: %.5f" % recall)
