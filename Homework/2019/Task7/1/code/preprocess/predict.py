import sys
import time

try:
    filename = sys.argv[1]
    task_id = sys.argv[2]
    server_ip = sys.argv[3]
    server_port = sys.argv[4]
except:
    print("Parameters Are Missing"
          "\n\tUsage: python3 predict.py [filename] [task_id] [server_ip] [server_port]")
    sys.exit(0)

import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from socket import *
import json
import threading
from top import *
import os

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph("./models/models.meta")
saver.restore(sess, tf.train.latest_checkpoint("./models/"))
from log_parse import run


def run_send(res_dict, task_id, server_ip, server_port):
    port = server_port
    ip = server_ip

    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((ip, port))

    obj = []
    obj.append(task_id)
    obj.append(res_dict)

    json_obj = json.dumps(obj)
    json_obj = bytes(json_obj, encoding='utf-8')
    sender = threading.Thread(target=send, args=(clientSocket, json_obj,))
    sender.start()


def send(clientSocket, json_obj):
    try:
        clientSocket.send(json_obj)
        print("[*]Done")
    except:
        print("[*]Fail")
    finally:
        clientSocket.close()


time_start = time.clock()

run(filename)
path = filename.split("/")
if len(path) == 1:
    path = "."
else:
    result = "."
    for i in range(0, len(path) - 1):
        result += "/" + path[i]
    path = result
graph = tf.get_default_graph()
x_ = graph.get_tensor_by_name("inputs/input_data:0")
len_ = graph.get_tensor_by_name("inputs/seq_length:0")
softmax_ = graph.get_tensor_by_name("Dense/Softmax:0")

with open(path + "/x_rand.pickle", "rb")as f_x, open(path + "/len_rand.pickle", "rb") as f_l, open("standard.pickle",
                                                                                                   "rb") as f_s:
    x = pickle.load(f_x)
    x = x.astype(np.float32)
    seq_len = pickle.load(f_l)
    seq_len = seq_len.astype(np.int32)
    standard = pickle.load(f_s)
    standard = standard.astype(np.float32)
    x = x.transpose(0, 2, 1)
    valid_x = len(x)
    x = np.vstack((x, standard))
    x = x.reshape(-1, 6)

    scaler = MinMaxScaler()
    scaled_x = scaler.fit_transform(x)

    scaled_x = scaled_x.reshape(-1, 51, 6)
    scaled_x = scaled_x[0:valid_x]
result = sess.run(softmax_, feed_dict={x_: scaled_x, len_: seq_len})
print(result)

result.tolist()
report = []
with open(path + "/log_path_seq_.csv", "r") as f:
    for session in result:
        line = f.readline()
        paths = line.strip().split(",")
        report.append((paths, (round(session[0], 4), round(session[1], 4))))

root_dic = os.path.abspath(path)  # 日志根目录
log_file = os.path.abspath(filename)
total_log = os.popen("wc -l " + filename).read().split(' ')[0]
total_sess = len(seq_len)
top_10_ip = top_ip(filename, 10)
top_5_path = top_path(path, 5)
total_anamoly_sess = list(filter(lambda x: x[1][1] > 0.5, report))
trans_sesses = list(map(lambda x: [y.split("/")[-1].split("?")[0] for y in x[0]], total_anamoly_sess))
anamoly_file = []

for s in trans_sesses:
    count = {}
    for p in s:
        try:
            count[p] += 1
        except:
            count[p] = 1
    count = count.items()
    count = sorted(count, key=lambda x: (x[1], x[0]), reverse=True)
    anamoly_file.append(count[0][0])
anamoly_file = list(set(anamoly_file))

send_dic = {
    "root_dic": root_dic,
    "log_file": log_file,
    "total_log": total_log,
    "total_sess": total_sess,
    "top_10_ip": top_10_ip,
    "top_5_path": top_5_path,
    "total_anamoly_sess": len(total_anamoly_sess),
    "anamoly_sess": total_anamoly_sess,
    "anamoly_file": anamoly_file
}
# run_send(send_dic, task_id, server_ip, server_port)

time_stop = time.clock()

print(time_stop - time_start)

print(send_dic)
