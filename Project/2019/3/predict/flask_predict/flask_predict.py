import lightgbm as lgb
import numpy as np
import pandas as pd
import _pickle as cPickle
from util import log,log_shape


file_name="flask_predict.csv"
model_name="lgb-0.962352(r34).txt"
chunkSize = 100000
column_values=['ip_app_channel_device_os_next_time_delta','ip_os_device_next_time_delta',
'ip_os_device_app_next_time_delta','ip_channel_prev_time_delta',
 'ip_os_prev_time_delta','nunique_channel_gb_ip',
 'nunique_app_gb_ip_device_os','nunique_hour_gb_ip_day',
 'nunique_app_gb_ip','nunique_os_gb_ip_app','nunique_device_gb_ip',
 'nunique_channel_gb_app','cumcount_os_gb_ip',
 'cumcount_app_gb_ip_device_os','count_gb_ip_day_hour','count_gb_ip_app',
 'count_gb_ip_app_os','var_day_gb_ip_app_os','ip','app','device','os',
 'channel','hour']

dtypes = {
    'click_id': 'uint32',
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8'
}


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


def convert_date(df):
    format = '%Y-%m-%d %H:%M:%S'
    df['date'] = pd.to_datetime(df['click_time'], format=format)
    return df


def preprocess_predict_data(test):
    log("start process predict data")
    save_test_supplement=cPickle.load(open('temp_test_supplement.p', 'rb'))
    temp_test_supplement=save_test_supplement.iloc[:1]
    test = test.merge(temp_test_supplement, left_on=['ip', 'app', 'device', 'os', 'channel', 'click_time'], right_on=['ip', 'app', 'device', 'os', 'channel', 'date'], how='left')
    test.drop(['click_time'], axis=1, inplace=True)
    predict=test[column_values]
    log("predict data load and process done")
    return predict


def load_predict_csv():
    log("start load predict data")
    test_reader=pd.read_csv(file_name, header=0, sep=',', dtype=dtypes, usecols=['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time'], parse_dates=['click_time'],iterator=True)
    test=read_csv_on_batch(test_reader,"test.csv")
    log("load predict data done!")

    return test

def lgb_predict(model,predict_feature):
    predict = model.predict(predict_feature)
    return predict

def load_model():
    model = lgb.Booster(model_file=model_name)
    return model

    
def main():
    raw_predict_data=load_predict_csv()
    predict_feature=preprocess_predict_data(raw_predict_data)
    model=load_model()
    predict_list=lgb_predict(model,predict_feature)
    n_predict_list=len(predict_list)
    n_predict_feature=len(predict_feature)
    print("n_predict_feature:{}".format(n_predict_feature))
    print("n_predict_list:{}".format(n_predict_list))
    click_ad=0
    for i in range(n_predict_feature):
        print("click_ad_index:{},is_attributed_prob:{}".format(i,predict_list[i]))
        if predict_list[i]>=0.75:
            print("疑似广告欺诈：click_ad_index:{},is_attributed_prob:{}".format(i,predict_list[i]))
            click_ad=click_ad+1
    print("###########总共有{}次点击为疑似广告欺诈########".format(click_ad))

    
if __name__ == "__main__":
    main()
    









