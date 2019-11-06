import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

MODEL_NAME='model'

def load_data(name):
    featureSet=pd.read_csv(name)
    X = featureSet.drop(['sql','label'],axis=1).values
    y = featureSet['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)
    return X,y,X_train, X_test, y_train, y_test


def train(X_train,y_train):
    clf=SVC()
    clf.fit(X_train,y_train)
    return clf

def save_model(clf,score):
    score=str(score)
    SAVE_MODEL_NAME=MODEL_NAME+'_'+score+'.pkl'
    joblib.dump(clf,SAVE_MODEL_NAME)

def evaluate_model(clf,X_test,y_test,X,y):
    score = clf.score(X_test,y_test)
    print ("###### Model score: {} ######".format(score))
    res = clf.predict(X)
    mt = confusion_matrix(y, res)
    print("###### False positive rate : %f %% ######" % ((mt[0][1] / float(sum(mt[0])))*100))
    print('###### False negative rate : %f %% ######' % ( (mt[1][0] / float(sum(mt[1]))*100)))
    return score

def main():
    X,y,X_train, X_test, y_train, y_test=load_data("feature.csv")
    clf=train(X_train,y_train)
    score=evaluate_model(clf,X_test,y_test,X,y)
    save_model(clf,score)

if __name__ == "__main__": 
    main()

    


