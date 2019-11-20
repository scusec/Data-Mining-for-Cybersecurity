# encoding=utf-8

# @Time:2019/2/12 16:19
# @Author:wyx
# @File:log_pre2.py

import re
import csv
import datetime
import pickle
import numpy as np
import path_relation
import referrer_relation

sess = []

user_ip = []
user_agent = []
users = []
COMBINED_LOGLINE_PAT = None
BOT_TRACES = None


def gen_user_time_seq(user):
    single_time_seq = []
    ops = user['op']
    for i in range(len(ops) - 1):
        b = datetime.datetime.strptime(ops[i + 1]['datetime'], '%d/%b/%Y:%H:%M:%S')
        a = datetime.datetime.strptime(ops[i]['datetime'], '%d/%b/%Y:%H:%M:%S')
        single_time_seq.append(abs((b - a).total_seconds()))
    single_time_seq.append(0)

    return single_time_seq


def gen_time_seq():
    time_seq = []
    for user in sess:
        single_time_seq = []
        ops = user['op']
        for i in range(len(ops) - 1):
            b = datetime.datetime.strptime(ops[i + 1]['datetime'], '%d/%b/%Y:%H:%M:%S')
            a = datetime.datetime.strptime(ops[i]['datetime'], '%d/%b/%Y:%H:%M:%S')
            single_time_seq.append(int(abs((b - a).total_seconds())))
        single_time_seq.append(0)
        time_seq.append(single_time_seq)
    return time_seq


def gen_seq():
    global sess
    method_seq = []
    byte_seq = []
    status_seq = []
    path_seq = []
    referrer_seq = []
    for user in sess:
        single_method_seq = []
        single_byte_seq = []
        single_status_seq = []
        single_path_seq = []
        single_referrer_seq = []
        for single in user['op']:
            # 生成每个用户方法的序列
            method = single['method']
            if method == 'POST':
                method = 1
            else:
                method = 0
            single_method_seq.append(method)

            # 生成每个用户的访问流量序列
            single_byte_seq.append(single['bytes'])

            # 生成每个用户的访问状态码序列
            single_status_seq.append(single['status'])

            # 生成每个用户的路径序列
            path = single['path']
            index = path.find('?')
            if index != -1:
                path = path[:index]
            single_path_seq.append(path)

            # 生成每个用户的referrer序列
            single_referrer_seq.append(single['referrer'])

        method_seq.append(single_method_seq)
        byte_seq.append(single_byte_seq)
        status_seq.append(single_status_seq)
        path_seq.append(single_path_seq)
        referrer_seq.append(single_referrer_seq)

    return method_seq, byte_seq, status_seq, path_seq, referrer_seq


def clear():
    global users
    new_users = []

    for user in users:
        single_time_seq = []
        ops = user['op']
        ip = user['ip']
        ua = user['client']
        for i in range(len(ops) - 1):
            b = datetime.datetime.strptime(ops[i + 1]['datetime'], '%d/%b/%Y:%H:%M:%S')
            a = datetime.datetime.strptime(ops[i]['datetime'], '%d/%b/%Y:%H:%M:%S')
            single_time_seq.append(int(abs((b - a).total_seconds())))
        single_time_seq.append(0)

        if not sum(i >= 630 for i in single_time_seq):
            if any(single_time_seq):
                new_users.append(user)
        else:
            for i in range(len(single_time_seq)):
                if single_time_seq[i] >= 630:
                    if i != 0:
                        user = {}
                        op = []
                        user['ip'] = ip
                        user['client'] = ua
                        for j in range(i + 1):
                            op.append(ops[j])
                        user['op'] = op
                        new_users.append(user)
                    if i != len(single_time_seq):
                        user = {}
                        op = []
                        user['ip'] = ip
                        user['client'] = ua
                        for j in range(i + 1, len(single_time_seq)):
                            op.append(ops[j])
                        user['op'] = op
                        users.append(user)
                    break
    tmp_users = new_users[:]
    for user in tmp_users:
        seq = gen_user_time_seq(user)
        if sum(i <= 1 for i in seq) == len(seq):
            new_users.remove(user)

    return new_users


def trans2dict(log):
    global users
    global user_ip
    global user_agent
    origin = log['origin']
    client = log['client']

    if origin not in user_ip:
        if client not in user_agent:
            op = []
            user = {}

            user_ip.append(origin)
            user_agent.append(client)
            user['ip'] = origin
            user['client'] = client

            del log['origin']
            del log['client']

            op.append(log)
            user['op'] = op
            users.append(user)
    else:
        for user in users:
            if (user['ip'] == origin and user['client'] == client):
                # if len(user['op']) <= 50:
                del log['origin']
                del log['client']
                user['op'].append(log)
                break


def log_prepare(file, is_static):
    global COMBINED_LOGLINE_PAT
    global BOT_TRACES
    with open(file) as f:
        while 1:
            logline = f.readline()
            if not logline:
                break
            isFalse = False
            log = {}
            match_info = COMBINED_LOGLINE_PAT.match(logline)

            if match_info is None:
                continue
            # 检测是否是爬虫
            for pat in BOT_TRACES:
                if pat.match(match_info.group('client')):
                    isFalse = True
                    break
            # #检测是否为异常访问
            # if match_info.group('status') in is_anomaly_visit:
            #         isFalse=True
            # 检测是否为静态文件
            for word in is_static:
                if word in match_info.group('path'):
                    isFalse = True
            if not isFalse:
                log['status'] = match_info.group('status')
                log['origin'] = match_info.group('origin')
                log['referrer'] = match_info.group('referrer')
                log['bytes'] = match_info.group('bytes')
                log['client'] = match_info.group('client')
                log['datetime'] = match_info.group('datetime')
                log['path'] = match_info.group('path')
                log['method'] = match_info.group('method')
                trans2dict(log)


def run(filename):
    global sess
    # 连接mysql数据库
    global users
    global user_ip
    global user_agent
    global COMBINED_LOGLINE_PAT
    global BOT_TRACES
    # list1 = ['400', '404', '405']
    list2 = ['ico', 'txt', 'html']
    path = filename.split("/")
    if len(path) == 1:
        path = "."
    else:
        result = "."
        for i in range(0, len(path) - 1):
            result += "/" + path[i]
        path = result

    # 提取字段
    COMBINED_LOGLINE_PAT = re.compile(
        r'(?P<origin>\d+\.\d+\.\d+\.\d+) '
        + r'(?P<identd>-|\w*) (?P<auth>-|\w*) '
        + r'\[(?P<datetime>[^\[\]:]+:\d+:\d+:\d+) (?P<tz>[\-\+]?\d\d\d\d)\] '
        + r'"(?P<method>\w+) (?P<path>[\S]+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<bytes>-|\d+)'
        + r'( (?P<referrer>"[^"]*")( (?P<client>"[^"]*")( (?P<cookie>"[^"]*"))?)?)?\s*\Z'
    )

    # spider黑名单
    BOT_TRACES = [
        (re.compile(r".*\+https://app\.hypefactors\.com/media-monitoring/about\.html.*")),
        (re.compile(r".*\+http://www\.semrush\.com/bot\.html.*")),
        (re.compile(r".*\+http://www\.google\.com/bot\.html.*")),
        (re.compile(r".*\+http://www\.baidu\.com/search/spider\.html.*")),
        (re.compile(r".*http://www\.bing\.com/bingbot\.htm.*")),
        (re.compile(r".*http://www\.opensiteexplorer\.org/dotbot.*")),
        (re.compile(r".*\+https://code\.google\.com/p/feedparser/.*")),
        (re.compile(r".*https://github\.com/Athou/commafeed.*")),
        (re.compile(r".*\+http://www.sogou.com/docs/help/webmasters.htm.*")),
        (re.compile(r".*\+https://t\.me/RSSchina_bot.*")),
        (re.compile(r".*\+http://yandex\.com/bots.*")),
        (re.compile(r".*YisouSpider/.*")),
        (re.compile(r".*DNSPod-Monitor/.*")),
        (re.compile(r".*SEMrushBot.*")),
        (re.compile(r".*http://www\.uptimerobot\.com/.*")),
        (re.compile(r".*\+http://www.feedly.com/fetcher.html.*")),
        (re.compile(r".*\+http://naver\.me/spd.*")),
        (re.compile(r".*\+https://api\.slack\.com/robots.*")),
        (re.compile(r".*http://mj12bot\.com/.*")),
        (re.compile(r".*\+https://t\.me/RustRssBot.*")),
        (re.compile(r".*http://www\.uptimerobot\.com.*")), ]

    # for i in range(1, 53):
    user_ip = []
    user_agent = []
    users = []
    log_prepare(filename, list2)
    # 细分会话
    new_users = clear()
    sess.extend(new_users)

    method_seq, byte_seq, status_seq, path_seq, referrer_seq = gen_seq()
    time_seq = gen_time_seq()

    with open(path + '/log_method_seq_.csv', 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerows(method_seq)

    with open(path + '/log_byte_seq_.csv', 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerows(byte_seq)

    with open(path + '/log_path_seq_.csv', 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerows(path_seq)

    with open(path + '/log_status_seq_.csv', 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerows(status_seq)

    with open(path + '/log_referrer_seq_.csv', 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerows(referrer_seq)

    with open(path + '/log_time_seq_.csv', 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerows(time_seq)
    path_relation.run(path + '/log_path_seq_.csv')
    referrer_relation.run(path + '/log_referrer_seq_.csv')
    preprocess(path)


def preprocess(path):
    x_seq = []
    len_seq = []
    longest = 51
    with open(path + "/log_byte_seq_.csv", "r") as f1, open(path + "/log_method_seq_.csv", "r") as f2, open(
            path + "/log_path_seq_processed.csv", "r") as f3, open(path + "/log_referrer_seq_processed.csv",
                                                                   "r") as f4, open(
        path + "/log_status_seq_.csv", "r") as f5, open(path + "/log_time_seq_.csv", "r") as f6:
        l1 = [int(x) for x in f1.readline().split(",")]
        l2 = [int(x) + 1 for x in f2.readline().split(",")]
        l3 = [int(x) + 1 for x in f3.readline().split(",")]
        l4 = [int(x) + 1 for x in f4.readline().split(",")]
        l5 = [int(x) for x in f5.readline().split(",")]
        l6 = [int(x) for x in f6.readline().split(",")]
        valid_len = len(l1)
        if valid_len > longest:
            l1 = l1[:51]
            l2 = l2[:51]
            l3 = l3[:51]
            l4 = l4[:51]
            l5 = l5[:51]
            l6 = l6[:51]
        len_id = 1
        while True:
            if valid_len < 51:
                l1 = np.pad(l1, (0, longest - valid_len), 'constant')
                l2 = np.pad(l2, (0, longest - valid_len), 'constant')
                l3 = np.pad(l3, (0, longest - valid_len), 'constant')
                l4 = np.pad(l4, (0, longest - valid_len), 'constant')
                l5 = np.pad(l5, (0, longest - valid_len), 'constant')
                l6 = np.pad(l6, (0, longest - valid_len), 'constant')

            if len(l1) != longest or len(l2) != longest or len(l3) != longest or len(l4) != longest or len(
                    l5) != longest or len(l6) != longest:
                pass
            else:
                x_seq.append([l1, l2, l3, l4, l5, l6])
                len_seq.append(valid_len)
                len_id += 1
            try:
                l1 = [int(x) for x in f1.readline().split(",")]
                l2 = [int(x) + 1 for x in f2.readline().split(",")]
                l3 = [int(x) + 1 for x in f3.readline().split(",")]
                l4 = [int(x) + 1 for x in f4.readline().split(",")]
                l5 = [int(x) for x in f5.readline().split(",")]
                l6 = [int(x) for x in f6.readline().split(",")]
                valid_len = len(l1)
            except:
                break
    x = np.array(x_seq)
    l = np.array(len_seq)
    with open(path + "/x_rand.pickle", "wb") as out_1, open(path + "/len_rand.pickle", "wb") as out_3:
        pickle.dump(x, out_1)
        pickle.dump(l, out_3)
