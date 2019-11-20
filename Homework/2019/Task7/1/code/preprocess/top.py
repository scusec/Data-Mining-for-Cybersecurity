import re


def top_path(path, num):
    with open(path + "/log_path_seq_.csv", "r") as f:
        dic = {}
        line = f.readline()
        while line:
            splited = line.strip().split(",")
            for ip in splited:
                try:
                    dic[ip] += 1
                except:
                    dic[ip] = 1
            line = f.readline()
    dic = dic.items()
    dic = sorted(dic, key=lambda x: (x[1], x[0]), reverse=True)
    return dic[:num] if len(dic) > num else dic


def top_ip(filename,num):
    with open(filename, "r") as f:
        dic = {}
        line = f.readline()
        while line:
            ip = line.split(" ")[0]
            try:
                dic[ip] += 1
            except:
                dic[ip] = 1
            line = f.readline()
    dic = dic.items()
    dic = sorted(dic, key=lambda x: (x[1], x[0]), reverse=True)
    return dic[:num] if len(dic) > num else dic
