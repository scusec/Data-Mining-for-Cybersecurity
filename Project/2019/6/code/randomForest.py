import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from random import shuffle
import joblib

df = pd.read_csv("./data/features.csv")
df = df.sample(frac=1).reset_index(drop=True)

X = df.drop(['label'], axis=1).values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier(n_estimators=100, oob_score=True)
print("开始训练…………")
clf.fit(X_train, y_train)
print("训练结束！")
y_pre = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pre)
recall = recall_score(y_test, y_pre, average="weighted")
joblib.dump(clf, './Model/randomForest.pkl')
print("模型保存成功！")

print("Accuracy: %.5f" % accuracy)
print("Recall: %.5f" % recall)



