import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from create_feature import getFeatures

result = pd.DataFrame(columns=('payload','dots','\"','\'','java',\
'script','alert','%','<','>','style','iframe','97','108','\x60','&#x28','label'))

def load_model(model_name):
    clf=joblib.load(model_name)
    return clf

def predict(clf,payload):
    global result
    results = getFeatures(payload,'')
    result.loc[0] = results
    result = result.drop(['payload','label'],axis=1).values
    print(clf.predict(result))

def main():
    model_name=input("Please input your model name:")
    payload=input("Please input your payload:")
    clf=load_model(model_name)
    predict(clf,payload)

if __name__ == "__main__":
    main()




