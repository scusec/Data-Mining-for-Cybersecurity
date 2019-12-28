from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
import _pickle as cPickle
import time
import datetime
import math
import gc
import warnings

warnings.filterwarnings('ignore')

dtypes = {
    'click_id': 'uint32',
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8'
}

train_len=0
predictors = []
chunkSize = 100000

#分批读取文件存储在chunks中，返回df形式
def read_csv_on_batch(reader,file_name):
    loop = True
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Reading {} is done".format(file_name))
    df = pd.concat(chunks, ignore_index=True)
    return df

#
def merge_count(df, columns_groupby, new_column_name, type='uint64'):
    add = pd.DataFrame(df.groupby(columns_groupby).size()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_nunique(df, columns_groupby, column, new_column_name, type='uint64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].nunique()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_cumcount(df, columns_groupby, column, new_column_name, type='uint64'):
    df[new_column_name] = df.groupby(columns_groupby)[column].cumcount().values.astype(type)
    predictors.append(new_column_name)
    return df


def merge_sum(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].sum()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    # predictors.append(new_column_name)  # bug: twice
    return df


def merge_var(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].var()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_rank(df, columns_groupby, column, new_column_name, ascending=True, type='uint64'):
    df[new_column_name] = df.groupby(columns_groupby)[column].rank(ascending=ascending)
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))


def log_shape(train, test):
    log('Train data shape: %s' % str(train.shape))
    log('Test data shape: %s' % str(test.shape))


def process_date(df):
    format = '%Y-%m-%d %H:%M:%S'
    df['date'] = pd.to_datetime(df['click_time'], format=format)
    df['month'] = df['date'].dt.month.astype('uint8')
    df['weekday'] = df['date'].dt.weekday.astype('uint8')
    df['day'] = df['date'].dt.day.astype('uint8')
    df['hour'] = df['date'].dt.hour.astype('uint8')
    df['minute'] = df['date'].dt.minute.astype('uint8')
    df['second'] = df['date'].dt.second.astype('uint8')
    df['tm_hour'] = (df['hour'] + df['minute'] / 60.0).astype('float32')
    df['tm_hour_sin'] = (df['tm_hour'].map(lambda x: math.sin((x - 12) / 24 * 2 * math.pi))).astype('float32')
    df['tm_hour_cos'] = (df['tm_hour'].map(lambda x: math.cos((x - 12) / 24 * 2 * math.pi))).astype('float32')
    del df['click_time']
    return df


def cal_next_time_delta(df, suffix, type='float32'):
    groupby_columns = [
        {'columns': ['ip', 'app', 'channel', 'device', 'os']},
        {'columns': ['ip', 'os', 'device']},
        {'columns': ['ip', 'os', 'device', 'app']}
    ]
    # Calculate the time to next click for each group
    for spec in groupby_columns:
        # Name of new feature
        new_name = '{}_{}'.format('_'.join(spec['columns']), suffix)
        # Unique list of features to select
        all_features = spec['columns'] + ['date']
        # Run calculation
        log('Calculate ' + suffix + '...')
        df[new_name] = (df[all_features].groupby(spec['columns']).date.shift(-1) - df.date).dt.seconds.astype(type)
        predictors.append(new_name)
        gc.collect()
    return df


def cal_prev_time_delta(df, suffix, type='float32'):
    groupby_columns = [
        {'columns': ['ip', 'channel']},
        {'columns': ['ip', 'os']}
    ]
    # Calculate the time to prev click for each group
    for spec in groupby_columns:
        # Name of new feature
        new_name = '{}_{}'.format('_'.join(spec['columns']), suffix)
        # Unique list of features to select
        all_features = spec['columns'] + ['date']
        # Run calculation
        log('Calculate ' + suffix + '...')
        df[new_name] = (df.date - df[all_features].groupby(spec['columns']).date.shift(+1)).dt.seconds.astype(type)
        predictors.append(new_name)
        gc.collect()
    return df


