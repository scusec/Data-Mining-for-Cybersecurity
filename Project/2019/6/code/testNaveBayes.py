import joblib
import numpy as np
import pandas as pd
from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
from getPermission import getPermissions
from naveBayes import setOfWords2Vec


def testNB(testFile):
    permissions = getPermissions(testFile)
    file = open("vacablaryList.txt", "r", newline="")
    vacablaryList = file.read().splitlines()
    vector = setOfWords2Vec(vacablaryList, permissions)
    vector = np.array(vector).reshape(1, -1)
    model = joblib.load("./Model/NB.pkl")
    result = model.predict(vector)
    pro = model.predict_proba(vector)
    file.close()
    return result, pro

if __name__ == "__main__":
    testFile = "VirusShare_00555e17bb04fe24c3cf34bac923f98a"
    result, pro = testNB(testFile)