from flask import Flask, render_template, request
import os
import pandas as pd
import pickle
from math import log
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)

cur_dir = os.path.dirname('__file__')
forest = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'forest.pkl'), 'rb'))

def pcap_process(file):
    try:
        os.system('tshark -r ' + file + ' -T fields -Y "dns" -e frame.number -e frame.time_relative -e ip.src -e ip.dst -e udp.length -e udp.srcport -e udp.dstport -e dns.flags.response -e dns.flags.rcode -e dns.qry.name -e dns.qry.name.len -e dns.count.labels -e dns.qry.type -e dns.qry.class -e dns.count.queries -e dns.count.answers -e dns.count.auth_rr -e dns.count.add_rr -e dns.resp.name -e dns.resp.len -e dns.resp.ttl -E separator="," -E aggregator=" " -E header=y -E occurrence=f > out.csv')
    except:
        print('error in pcap process')

def count_type(x):
    ret = 0
    for qry in x:
        if qry != 1 and qry != 5:
            ret += 1
    return 100 * ret / x.size

def calc_entropy(x):
    numEntries = x.size
    labelCounts = {}
    for label in x:
        if label not in labelCounts.keys(): labelCounts[label] = 0
        labelCounts[label] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def data_process(df):
    def swap_column(df, l, r):
        tmp = df.pop(l)
        df[l] = df[r]
        df[r] = tmp

    def qry_process(df_qry):
        df_qry.insert(9, 'dns.qry.subname', df_qry['dns.qry.name'].apply(lambda x: '.'.join(x.split('.')[-2:]) if type(x) == str else ''))
        df_qry.insert(10, 'dns.qry.subname_len', df_qry['dns.qry.subname'].apply(lambda x: len(x)))
        df_qry['udp.length'] = df_qry['udp.length'] - 8
        df_qry.rename(columns={'udp.length': 'dns.length'}, inplace=True)
        gp = df_qry.groupby(['ip.src'])
        newdf = pd.DataFrame(gp.agg({'dns.length': 'mean'}))  # DNS总长度均值
        newdf = newdf.join(gp.agg({'dns.qry.name.len': 'sum'}))
        newdf = newdf.join(gp.agg({'dns.qry.name.len': ['mean', 'max']}), lsuffix='qryLen')  # 请求域名均值与最大值
        newdf = newdf.join(gp.agg({'dns.count.labels': 'mean'}))  # 域名标签数量，标签指点分割的两段
        newdf = newdf.join(gp.agg({'dns.qry.type': count_type}))
        newdf.rename(columns={'dns.qry.type': 'typeNot_qryCount'}, inplace=True)
        newdf = newdf.join(gp.agg({'dns.qry.subname_len': ['max', 'mean']}), lsuffix='subname_len')
        newdf = newdf.join(gp['dns.qry.name'].agg('nunique'))  # 不同主机名的个数
        newdf = newdf.join(gp['dns.qry.subname'].agg(calc_entropy))
        newdf.rename(columns={'dns.qry.subname': 'dns.qry.subname_entropy'}, inplace=True)
        return newdf

    def res_process(df_res):
        swap_column(df_res, 'ip.src', 'ip.dst')
        swap_column(df_res, 'udp.srcport', 'udp.dstport')
        gp = df_res.groupby(['ip.src'])
        newdf = pd.DataFrame(gp.agg({'dns.resp.len': 'mean'}))  # 响应数据的长度均值
        newdf = newdf.join(gp.agg({'dns.resp.ttl': 'mean'}))  # 响应TTL的均值
        newdf = newdf.join(gp.agg({'dns.count.queries': 'mean'}))  # 请求中请求数量的均值
        newdf = newdf.join(gp.agg({'dns.count.answers': 'mean'}))  # 请求中响应数量的均值
        return newdf

    df_qry = qry_process(df.loc[df['udp.srcport'] != 53])  # 源端口不是53的归为请求
    df_res = res_process(df.loc[(df['udp.srcport'] == 53) & (df['dns.flags.response'] == 1)])  # 源端口是53且是响应数据包的归为响应

    newdf = df_qry.join(df_res)
    newdf = newdf.fillna(0)

    return newdf

def preprocess(file):
    data = pd.read_csv(file)
    data_processed = data_process(data)
    return data_processed

def classify(data):
    data_processed = preprocess(data)
    y_pred = forest.predict(data_processed)
    result = []
    for y in y_pred:
        result.append(y)
    data_processed['label'] = result
    data_processed.to_csv('result.csv', encoding='utf-8')

    Data = pd.read_csv('result.csv')
    output = ''
    for i in range(Data.shape[0]):
        if Data['label'][i] == 1:
            output = output + Data['ip.src'][i] + ": BotNet\n"

    if len(output) == 0:
        output = 'No BotNet in your pcap!'
    return output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(cur_dir, 'uploads', f.filename))
        pcap_process(os.path.join(cur_dir, 'uploads', f.filename))
        result = classify('out.csv')
        return render_template('index.html',
                              content_upload=f.filename,
                              prediction_upload=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()