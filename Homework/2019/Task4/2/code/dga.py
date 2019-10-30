#%%
#import all deps.
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

vowel_letters = ['a', 'e', 'i', 'o', 'u']

#%%
# Get orignal data.
alexa_dataset = pd.read_csv(
    r"C:\Users\Devin Wang\Desktop\DGA\alexa.csv", encoding="utf-8")
dga_dataset = pd.read_csv(
    r"C:\Users\Devin Wang\Desktop\DGA\all_dga.txt", encoding="utf-8", sep=' ')
X_1 = dga_dataset.iloc[:, 0].values
Y_1 = np.ones(X_1.shape[0])
X_2 = alexa_dataset.iloc[:, 1].values
Y_2 = np.zeros(X_2.shape[0])
X = np.append(X_1, X_2)
Y = np.append(Y_1, Y_2)


# %%
# 这部分是辅助函数：
def num_vowels(domain):
    num = 0
    for letter in domain.lower():
        if is_vowel(letter):
            num += 1
    return num


def is_num(char):
    if char >= '0' and char <= '9':
        return True
    return False


def is_vowel(char):
    letter = char.lower()
    if letter in vowel_letters:
        return True
    return False


def get_constant_consonant_list(domain):
    cons_list = []
    conso = re.finditer(r'([bcdfghjklmnpqrstvwxyz])*', domain.lower())
    for node in conso:
        if len(node.group()) > 1:
            cons_list.append(node.group())
    return cons_list


def get_freq_dict(domain):
    freq_dict = dict(Counter(domain))
    return freq_dict


# N-gram
def psb_n_gram(domain_list, n=2):
    domain_list = np.array(domain_list)
    CV = CountVectorizer(ngram_range=(1, 1))
    return CV.fit_transform(domain_list)


#%%
# 这部分是特征抽取函数:
# 1.domain长度
def length_of(domain):
    return len(str(domain))


# 2.元音字母占全部字符的比例 - 元音特征
def vowel_letter_ratio(domain):
    return (float)(num_vowels(domain) / length_of(domain))


# 3.连续的辅音(串数量)占全部字符的比例 - 辅音特征
def constant_consonant_ratio(domain):
    return (float)(
        len(get_constant_consonant_list(domain)) / length_of(domain))


# 4.数字占全部字符比例 - 数字特征
def number_ratio(domain):
    count = 0
    for letter in domain:
        if is_num(letter):
            count += 1
    return (float)(count / length_of(domain))


# 5.Domain信息熵
def calc_entropy(domain):
    ent = 0
    l = length_of(domain)
    all_letters = dict(Counter(domain)).keys()
    freq_dict = get_freq_dict(domain)
    for letter in all_letters:
        frequency = freq_dict[letter]
        ent -= (frequency / l) * np.log2(frequency / l)
    return ent


# 6.Domain中0-9、a-f总长度的比例
def count_hex_digit_words_ratio(domain):
    hex_count = 0
    for letter in domain.lower():
        if (letter >= 'a' and letter <= 'f') or is_num(letter):  #是hex digit
            hex_count += 1
    return (float)(hex_count / len(domain))


# 7.唯一出现的字母占所有出现过的字母的比例
def count_unique_letter_ratio(domain):
    domain = domain.lower()
    unq_count = 0
    freq_dict = get_freq_dict(domain)
    for value in freq_dict.values():
        if value == 1:
            unq_count += 1
    return (float)(unq_count / len(freq_dict))


# 8.TLD顶级域检测
def tld_is_com_or_cn(url):
    slices = url.lower().split('.')
    if slices[-1] == 'com' or slices[-1] == 'cn':
        return 1
    return 0


#%%
# Get features.
vectors = np.full((X.shape[0], 9), 0, dtype=np.float)
for i in range(X.shape[0]):
    vectors[i, 0] = length_of(X[i])
    vectors[i, 1] = vowel_letter_ratio(X[i])
    vectors[i, 2] = constant_consonant_ratio(X[i])
    vectors[i, 3] = number_ratio(X[i])
    vectors[i, 4] = calc_entropy(X[i])
    vectors[i, 5] = count_hex_digit_words_ratio(X[i])
    vectors[i, 6] = count_unique_letter_ratio(X[i])
    vectors[i, 7] = tld_is_com_or_cn(X[i])
    vectors[i, 8] = Y[i]
vectors = shuffle(vectors)
encoded_X = vectors[:, 0:8]
encoded_Y = vectors[:, 8]
scaler = StandardScaler()
encoded_X = scaler.fit_transform(encoded_X)


#%%
# Offer a callback impl.
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)


metrics = Metrics()

#%%
# Create model
model = Sequential()
model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

#%%
# Fit.
pres = []
recalls = []
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(encoded_X, encoded_Y):
    train_X, test_X = encoded_X[train_index], encoded_X[test_index]
    train_Y, test_Y = encoded_Y[train_index], encoded_Y[test_index]
    model.fit(
        train_X,
        train_Y,
        batch_size=512,
        epochs=10,
        callbacks=[metrics],
        validation_data=(test_X, test_Y))
    pre = sum(metrics.val_precisions) / len(metrics.val_precisions)
    recall = sum(metrics.val_recalls) / len(metrics.val_recalls)
    pres.append(pre)
    recalls.append(recall)
    print("Precision: {}%. Recall: {}%".format(pre * 100.0, recall * 100.0))

# %%
ave_pre = sum(pres) / len(pres)
ave_recall = sum(recalls) / len(recalls)
print("Precision: {}%. Recall: {}%".format(ave_pre * 100.0,
                                           ave_recall * 100.0))

# %%
from sklearn.tree import DecisionTreeClassifier
# for train_index, test_index in kfold.split(encoded_X, encoded_Y):
tree = DecisionTreeClassifier(random_state=None).fit(encoded_X,
                                                        encoded_Y )
prds = tree.predict(encoded_X)
pre = precision_score(encoded_Y, prds)
rec = recall_score(encoded_Y, prds)
print("Precision: {}. Recall: {}.".format(pre, rec))


# %%
