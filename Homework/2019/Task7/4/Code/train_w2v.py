# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import getData
from gensim.models import word2vec

num_features = 120    # Num of word vector dimenstion
min_word_count = 30   # Minimum word count
num_workers = 8       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3
webshell_json = "./data/webshell.json"
normal_json = "./data/normal.json"
model_name = "./data/120features_30minwords_10context"

print('[*]Trainning Word2Vector...')

train_set, labels = getData.getAllData(webshell_json, normal_json)

model = word2vec.Word2Vec(train_set, workers=num_workers,
                          size=num_features, min_count=min_word_count,
                          window=context, sample=downsampling)

model.save(model_name)

print("DONE")
