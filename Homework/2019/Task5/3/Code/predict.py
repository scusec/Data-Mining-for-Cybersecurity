import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from create_feature import getFeatures

result = pd.DataFrame(columns=('sql','num_f','capital_f','key_num','space_f','special_f','prefix_f','label'))


def load_model(model_name):
    clf=joblib.load(model_name)
    return clf

def predict(clf,sql):
    global result
    results = getFeatures(sql,'')
    result.loc[0] = results
    result = result.drop(['sql','label'],axis=1).values
    print(clf.predict(result))

def main():
    model_name=input("Please input your model name:")
    sql=input("Please input your sql:")
    clf=load_model(model_name)
    predict(clf,sql)

if __name__ == "__main__":
    main()




