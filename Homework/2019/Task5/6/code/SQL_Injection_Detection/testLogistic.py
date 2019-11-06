import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from data import dataGenerate
import csv

payload = input("请输入测试用的payload:")
testPayload = open("testFile.txt", "w")
testPayload.writelines(payload)
testPayload.close()
model = input("请输入选用的模型:")
clf = joblib.load("./Model/" + model)

result = pd.DataFrame(columns = ("entropy", "length" ,"num" ,"capitalPro" ,"evilWord" ,"maxLength" ,"label"))
results = dataGenerate("testFile.txt", "resultFile.csv", 1)
with open(results, "r", encoding = "utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for i,rows in enumerate(reader):
        if i == 0:
            row = rows
result.loc[0] = row
result = result.drop(["label"], axis = 1).values
print(clf.predict(result))
