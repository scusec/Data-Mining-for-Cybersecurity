import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from create_feature import getFeatures

result = pd.DataFrame(columns=('url','no of dots','presence of hyphen','len of url','presence of at',\
'presence of double slash','no of subdir','no of subdomain','len of domain','no of queries','is IP','presence of Suspicious_TLD',\
'presence of suspicious domain','label'))

def load_model(model_name):
    clf=joblib.load(model_name)
    return clf

def predict(clf,url):
    global result
    results = getFeatures(url,'')
    result.loc[0] = results
    result = result.drop(['url','label'],axis=1).values
    print(clf.predict(result))

def main():
    model_name=input("Please input your model name:")
    url=input("Please input your url:")
    clf=load_model(model_name)
    predict(clf,url)

if __name__ == "__main__":
    main()




