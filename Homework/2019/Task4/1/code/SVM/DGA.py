import csv
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,recall_score
import joblib
from sklearn.preprocessing import StandardScaler

goodlist = []
badlist = []


def length(domain_name):
    return len(domain_name)

def infoentropy(domain_name):
    charlist = []
    for char in domain_name:
        if char not in charlist:
            charlist.append(char)
    countlist = []
    for char in charlist:
        countlist.append(domain_name.count(char))
    result = 0
    for count in countlist:
        result += (-1 * count/length(domain_name) * math.log2(count/length(domain_name)))
    return result

def vowel(domain_name):
    vcount = 0
    domain_name = domain_name.split('.')
    maxlength, domain_name = max([(len(x),x) for x in domain_name])
    for char in domain_name:
        if char in ['a', 'e', 'i', 'o', 'u']:
            vcount += 1
    return vcount/length(domain_name)

def consonant(domain_name):
    ccount = 0
    for i in range(len(domain_name)):
        if domain_name[i] not in ['a', 'e', 'i', 'o', 'u'] and domain_name[i].isalpha() and i + 1 < len(domain_name):
            if domain_name[i + 1] not in ['a', 'e', 'i', 'o', 'u'] and domain_name[i].isalpha():
                ccount += 1
                i += 1
    return ccount

def num(domain_name):
    result = 0
    for char in domain_name:
        if char.isalnum():
            result += 1
    return result


with open("top-1m.csv") as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        good = row[1]
        goodlist.append([length(good), infoentropy(good), vowel(good), consonant(good), num(good)])
        if len(goodlist) == 10000:
            break

with open("dga.txt") as f:
    data = f.readlines()
    for row in data:
        bad = row.split()[1]
        badlist.append([length(bad), infoentropy(bad), vowel(bad), consonant(bad), num(bad)])


if __name__ == '__main__':

    list = badlist + goodlist[:10000]
    matrix = np.array(list)
    # bad = 0, good = 1
    label = []
    for i in range(len(badlist)):
        label.append(0)
    for i in range(10000):
        label.append(1)

    sc = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(matrix, label, test_size=0.1)

    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    print("start training:")
    svm = SVC(kernel="rbf", random_state=1, C=0.9,gamma="auto", max_iter=100000)
    svm.fit(X_train_std, y_train)

    joblib.dump(svm, "svm_model")

    y_pred = svm.predict(X_test_std)
    print("acc:{}".format(accuracy_score(y_test, y_pred)))
    print("recall:{}".format(recall_score(y_test, y_pred)))

