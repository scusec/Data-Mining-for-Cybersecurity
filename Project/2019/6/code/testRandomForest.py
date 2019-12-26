import joblib
import numpy as np
import pandas as pd
from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
from getFeatures import getMd5
from getFeatures import getfileComEntropy
from getFeatures import getSignature
from getFeatures import getSize


def testRF(testFile):
    size = getSize(testFile)
    md5 = getMd5(testFile)
    signature = getSignature(testFile)
    comentropy = getfileComEntropy(testFile)
    sample = pd.DataFrame(columns=('label', 'size', 'md5', 'signature', 'comentropy'))
    sample.loc[0] = ['1', str(size), md5,  signature, comentropy]
    sample = sample.drop(['label'], axis=1).values

    model = joblib.load("./Model/randomForest.pkl")
    result = model.predict(sample)
    pro = model.predict_proba(sample)
    return result, pro

if __name__ == "__main__":
    testFile = "VirusShare_00555e17bb04fe24c3cf34bac923f98a"
    result, pro = testRF(testFile)