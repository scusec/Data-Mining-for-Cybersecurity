import csv
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


# 返回.的个数
def sub_domain(rows):
    result = []
    for row in rows:
        result.append(row[0].split("/")[0].count('.'))
    return result


# bad为0，good为1
def get_labels(rows):
    result = []
    for row in rows:
        if row[1].startswith('bad'):
            result.append(0)
        else:
            result.append(1)
    return result


# 有IP为1，没IP为0
def having_ip_address(rows):
    re_exp = r"([0-9]{1,3}\.){3}[0-9]{1,3}"
    result = []
    for row in rows:
        if re.search(re_exp, row[0].split('/')[0]):
            result.append(0)
        else:
            result.append(0)
    return result


# 返回域名长度
def domain_length(rows):
    result = []
    for row in rows:
        result.append(len(row[0].split("/")[0]))
    return result


# 有@返回1，没有返回0
def having_alt_symbol(rows):
    result = []
    for row in rows:
        if row[0].find("@") < 0:
            result.append(0)
        else:
            result.append(1)
    return result


# 有#返回1，没有返回0
def having_anchor(rows):
    result = []
    for row in rows:
        if row[0].find("#") < 0:
            result.append(0)
        else:
            result.append(1)
    return result


def num_of_paraments(rows):
    result = []
    for row in rows:
        try:
            result.append(row[0].split("?")[1].count('&') + 1)
        except:
            result.append(0)
    return result


def having_port(rows):
    result = []
    for row in rows:
        if row[0].split("/")[0].find(":") < 0:
            result.append(1)
        else:
            result.append(0)
    return result


if __name__ == '__main__':
    with open('data.csv', 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        rows = []
        for row in reader:
            rows.append(row)
    rows = rows[1:]
    for index in range(len(rows)):
        try:
            rows[index][0] = rows[index][0].split("://")[1]
        except:
            continue

    matrix = [sub_domain(rows), having_ip_address(rows), domain_length(rows), having_alt_symbol(rows),
              having_anchor(rows), num_of_paraments(rows), having_port(rows)]

    final_data = np.transpose(matrix)

    labels = get_labels(rows)

    sc = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2)

    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    svm = SVC(kernel="rbf", random_state=1, C=0.9, gamma=0.2, max_iter=100000)
    svm.fit(X_train_std, y_train)

    joblib.dump(svm, "svm_model")

    y_pred = svm.predict(X_test_std)
    print(accuracy_score(y_test, y_pred))
