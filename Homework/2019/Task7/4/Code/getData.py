# -*- coding: utf-8 -*-
# 模拟已经有数据集的情况
# 需要使得两种数据混杂在一起
# 
import json
import random
import numpy as np

def load_json_file(filename):
    json_file = open(filename, "rb")
    json_obj = json_file.read()
    json_list = json.loads(json_obj)
    return json_list

# 原本的是opcode的一个list, 每个元素都是单个文件的opcode
# 这个函数将其变成列表的列表
def parse_list(raw_list):
    result = []
    for file_opc in raw_list:
        single_file_opcs = file_opc.split(' ')
        result.append(single_file_opcs)
    return result

# combined with opcodes
def parse_raw_opcodes(raw_list):
    return parse_list(raw_list)

# 整合两个组(webshell和正常组)
# 本次使用的数据集大概是 1711(webshell) : 4429(normal)
# 接近2.5的比例， 2 ： 5
# 0 -- webshell
# 1 -- webshell
# 2,3,4,5,6 -- normal

def scale(webshells, normals):
    data_set = []
    label = []
    
    web_len = len(webshells)
    nor_len = len(normals)
    
    # iterator
    # i for webshells
    # j for normals
    i = 0
    j = 0
    
    for i in range(web_len + nor_len):   
        die = random.randint(0,6)
        if (die <= 1):
            # reach the end
            if (i >= web_len):
                data_set.extend(normals[j:])
                label.extend([0 for x in range(j,nor_len)])
                break
            data_set.append(webshells[i])
            label.append(1)
            i += 1
        else:
            if(j >= nor_len):
                data_set.extend(webshells[i:])
                label.extend([1 for x in range(i,web_len)])
                break
            data_set.append(normals[j])
            label.append(0)
            j += 1
    
    return data_set, label

def getAllData(webshell_name, normal_name):
    # load opcode
    webshell = load_json_file(webshell_name)
    normal = load_json_file(normal_name)
    
    webshell_list = parse_list(webshell)
    normal_list = parse_list(normal)
    
    dataset,label = scale(webshell_list, normal_list)
    return dataset, label
    
# 总数是6140， 方便直接硬编码，取十分之一作为测试咯
def splitToTrainAndTest(dataset, label):
    test_set = dataset[:614]
    test_label = label[:614]
    train_set = dataset[614:]
    train_label = label[614:]
    
    return train_set, train_label, test_set, test_label

def getOneHot(labels):
    result = []
    for label in labels:
        if label == 1:
            result.append(list([1,0]))
        elif label == 0:
            result.append(list([0,1]))
    return np.array(result)
