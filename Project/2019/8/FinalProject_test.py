# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:30:47 2019

@author: Birdman
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import numpy as np
import time
import datetime
from bs4 import BeautifulSoup
import requests
import math
import re
import pandas as pd
from sklearn import preprocessing

np.warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)#取消科学计数法显示
np.set_printoptions(threshold = np.inf) #全部显示数据不用省略号代替

flags = tf.app.flags
FLAGS = flags.FLAGS



#结构体，服务于load_data()，使之返回特征和类标
class Rdata():
    def __init__(self, data, target, uid):
        self.data = data
        self.target = target
        self.uid = uid
    def make_struct(self, data, target, uid):
        return self.Struct(data, target, uid)


#读取csv文件，并返回数据和标签
class User_Info_Raw():
    def __init__(self, uid, fansnum, follownum, forwardnum, registertime, weibonum, weibotime):
        self.uid = uid
        self.fansnum = fansnum
        self.follownum = follownum
        self.forwardnum = forwardnum
        self.registertime = registertime
        self.weibonum = weibonum
        self.weibotime = weibotime
    def make_struct(self, uid, fansnum, follownum, forwardnum, registertime, weibonum, weibotime):
        return self.Struct(self, uid, fansnum, follownum, forwardnum, registertime, weibonum, weibotime)

def change_data(filename, ipt): 
    with open(filename,'rt') as raw_data:#打开指定路径下的文件并读取
        readers = pd.read_csv(raw_data, encoding="utf-8", header=0)#将csv文件存入numpy数组中
        data = readers[readers.columns[1:-1]].values
        data = np.insert(data, 0, values=ipt, axis=0)
        scaler = preprocessing.StandardScaler()
        data = scaler.fit_transform(data)
        #print(data[0])
        return (data[0])

def get_info(uid):
    try:    
        fans_header = {
        'Host': 'weibo.cn',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Accept-Encoding': 'gzip,deflate,br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cookie': '_T_WM=41416814899; SUB=_2A25w9Jc1DeRhGeBG6VAY9i7LzDmIHXVQFjl9rDV6PUJbkdAKLUrNkW1NRgrWs5siPVTYw_5O_tvSu1ivM3TD6khx; SUHB=079j-1SX87IJeZ; SCF=Ak9bK12NJTStQaC1QB-foiusA5LyihZzgxNgv87FQypJYVhivbIG00q6mxR2wyb0l4MCo8ffZVwwPXJv6B5qyyA.; SSOLoginState=1576068965',
        }
        register_header = {
        'Host': 'weibo.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Accept-Encoding': 'gzip,deflate,br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cookie': '_T_WM=41416814899; SUB=_2A25w9Jc1DeRhGeBG6VAY9i7LzDmIHXVQFjl9rDV6PUJbkdAKLUrNkW1NRgrWs5siPVTYw_5O_tvSu1ivM3TD6khx; SUHB=079j-1SX87IJeZ; SCF=Ak9bK12NJTStQaC1QB-foiusA5LyihZzgxNgv87FQypJYVhivbIG00q6mxR2wyb0l4MCo8ffZVwwPXJv6B5qyyA.; SSOLoginState=1576068965',
        }
        url = "https://weibo.cn/u/"+uid
        flag = 0
        fans_info_response = requests.get(url, headers=fans_header)
        soup_1 = BeautifulSoup(fans_info_response.content, 'html.parser')
        #爬取关注数量
        temp = '/'+uid+'/'+'follow'
        follownum_info = soup_1.find('a', attrs={'href': temp})
        follownum = str(follownum_info).split('[')[1].split(']')[0]
        #print("关注数量：",follownum)
    
        #爬取粉丝数量
        fansnum_info = soup_1.find('a',attrs={'href': '/'+uid+'/fans'})
        fansnum = str(fansnum_info).split('[')[1].split(']')[0]
        #print("粉丝数量：", fansnum)
    
        # 爬取发博数量，顺便得到转发数量
        weibonum_info = soup_1.find('span', attrs={'class': 'tc'})
        #print(soup_1)
        #print(str(weibonum_info))
        weibonum = str(weibonum_info).split('[')[1].split(']')[0]
        #print("发博数：", weibonum)
    
        #如果没有发博数，就不爬取发博时间
        if weibonum == '0':
            #print("发博数为0,不再爬取发博时间！")
            flag += 1
            weibo_total_time = "NULL"
    
        elif int(weibonum) >= 30:
            
            # 爬取原创微博网页
            original_weibo_response = requests.get(url + "?filter=1", headers=fans_header)
    
            # 爬取原创微博页数，进而爬取原创微博总量
            soup_3 = BeautifulSoup(original_weibo_response.content, 'html.parser')
            original_pagenum_info = soup_3.find(name="input", attrs={"type": "submit", "value": "跳页"})
            # 有的原创微博数少，可能不足1页
            original_pagenum = ""
            if original_pagenum_info is None:
                original_pagenum = 1
            else:
                original_pagenum = str(original_pagenum_info.parent).split('>')[-2].split('/')[1][:-2]
    
            #转到原创的最后一页，查看总共有多少个原创微博
            last_original_weibo_response = requests.get(url + "?filter=1"+"&page="+str(original_pagenum), headers=fans_header)
    
            #打开网页，计算最后一页有多少个微博
            soup_4 = BeautifulSoup(last_original_weibo_response.content, 'html.parser')
            last_weibonum_info = soup_4.find_all('div',attrs={"class": "c"})
            last_weibonum = len(last_weibonum_info) - 3
            forwardnum = int(weibonum) - 10* (int(original_pagenum)-1) - last_weibonum
    
            #先提取当前页面的发博时间
            weibo_time = ""
            weibo_total_time = ""
    
            weibo_page_num = math.ceil(int(weibonum) / 10)
    
            for j in range(1,4):
                #拼接新的url
                newline = url + "?page=" + str(j)
    
                fans_weibo_response = requests.get(newline, headers=fans_header)
                
                # 打开页面，使用beautifulsoup()解析
                soup_5 = BeautifulSoup(fans_weibo_response.content, 'html.parser')
                weibo_time_info = soup_5.find_all(name="span", attrs={"class": "ct"})
    
                # 最后需要将所有时间存储到weibo_time中
                weibo_time = ''
                for i in weibo_time_info:
                    temp = str(i).split(">")[1].split('来')[0][:-1]
                    if len(temp) == 12:
                        temp = "2019年" + temp
                    #print(temp, "长度为", len(temp))
                    weibo_time += "#"+ temp
    
                weibo_total_time += weibo_time
            flag += 1
        else :
            # 爬取原创微博网页
            original_weibo_response = requests.get(url + "?filter=1", headers=fans_header)
            
            # 爬取原创微博页数，进而爬取原创微博总量
            soup_3 = BeautifulSoup(original_weibo_response.content, 'html.parser')
            original_pagenum_info = soup_3.find(name="input", attrs={"type": "submit", "value": "跳页"})
            #有的原创微博数少，可能不足1页
            original_pagenum = ""
            if original_pagenum_info is None:
                original_pagenum = 1
            else:
                original_pagenum = str(original_pagenum_info.parent).split('>')[-2].split('/')[1][:-2]
    
            #转到原创的最后一页，查看总共有多少个原创微博
            last_original_weibo_response = requests.get(url + "?filter=1"+"&page="+ str(original_pagenum), headers=fans_header)
    
            #打开网页，计算最后一页有多少个微博
            soup_4 = BeautifulSoup(last_original_weibo_response.content, 'html.parser')
            last_weibonum_info = soup_4.find_all('div',attrs={"class": "c"})
            last_weibonum = len(last_weibonum_info) - 3
            forwardnum = int(weibonum) - 10* (int(original_pagenum)-1) - last_weibonum
    
            weibo_time = ""
            weibo_total_time = ""
    
            weibo_page_num = math.ceil(int(weibonum) / 10)
    
            for j in range(1,weibo_page_num+1):
                #拼接新的url
                newline = url + "?page=" + str(j)
                fans_weibo_response = requests.get(newline, headers=fans_header)
               
                # 打开页面，使用beautifulsoup()解析
                soup_6 = BeautifulSoup(fans_weibo_response.content, 'html.parser')
                weibo_time_info = soup_6.find_all(name="span", attrs={"class": "ct"})
    
                # 最后需要将所有时间存储到weibo_time中
                weibo_time = ''
                for i in weibo_time_info:
                    temp = str(i).split(">")[1].split('来')[0][:-1]
                    if len(temp) == 12:
                        temp = "2019年" + temp
                    #print(temp, "长度为", len(temp))
                    weibo_time += "#"+ temp
                weibo_total_time += weibo_time
    
            #print(weibo_total_time)
    
        #爬取微博注册时间
        #更改url，获取新的信息
        new_url = "https://weibo.com/" + uid + "/info?mod=pedit_more"
        register_info_response = requests.get(new_url, headers=register_header)
        
        soup_7 = BeautifulSoup(register_info_response.content, 'html.parser')
        register_info = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", soup_7.get_text()).group(0)
        
        fansnum = int(fansnum)
        follownum = int(follownum)
        #forwardnum = int(forwardnum)
        registertime = str(register_info)
        weibonum = int(weibonum)
        weibotime = str(weibo_total_time)
        print('数据爬取完成……')
        return User_Info_Raw(uid = uid, fansnum = fansnum, follownum = follownum, forwardnum = forwardnum, registertime = registertime, weibonum  = weibonum, weibotime = weibotime)
    except:
        return('Err')

class User_Info_Proc():
    def __init__(self, uid, fansnum, follownum, weibonum, forwardnum, weibotimestd, totalDivRegtime, forwardDivTotal, registertime):
        self.uid = uid
        self.fansnum = fansnum
        self.follownum = follownum
        self.weibonum = weibonum
        self.forwardnum = forwardnum
        self.weibotimestd = weibotimestd
        self.totalDivRegtime = totalDivRegtime
        self.forwardDivTotal = forwardDivTotal
        self.registertime = registertime
        
    def make_struct(self, uid, fansnum, follownum, weibonum, forwardnum, weibotimestd, totalDivRegtime, forwardDivTotal, registertime):
        return self.Struct(self, uid, fansnum, follownum, weibonum, forwardnum, weibotimestd, totalDivRegtime, forwardDivTotal, registertime)

def proc(iptinfo):
    iptuid = iptinfo.uid
    iptfansnum = iptinfo.fansnum
    iptfollownum = iptinfo.follownum
    iptforwardnum = iptinfo.forwardnum
    iptregistertime = iptinfo.registertime
    iptweibonum = iptinfo.weibonum
    iptweibotime = iptinfo.weibotime
    
    #print(iptinfo.uid, iptinfo.fansnum, iptinfo.follownum, iptinfo.forwardnum, iptinfo.registertime, iptinfo.weibonum, iptinfo.weibotime)
    
    #注册时间
    try:
        register2now=time.time()/(60*60*24)-time.mktime(time.strptime(iptregistertime, "%Y/%m/%d"))/(60*60*24)
    except:
        register2now=time.time()/(60*60*24)-time.mktime(time.strptime(iptregistertime, "%Y-%m-%d"))/(60*60*24)
    #平均发博间隔
    totalDivRegtime=iptweibonum/register2now
    #转发数占比
    forwardDivTotal = iptforwardnum/iptweibonum
    
    #发博时间方差
    timeBase = time.mktime(time.localtime(time.time()))
    rowSplit=iptweibotime.split('#')[1:]
    if len(rowSplit) == 0:
        weibotimestd = 0
    for i in range(0,len(rowSplit)):
        if( "分钟前" in rowSplit[i]):
            mintueBefore = re.sub("\D", "", rowSplit[i])
            formatTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeBase - float(mintueBefore)*60))
            rowSplit[i] = formatTime
        elif("今天" in rowSplit[i]):
            dayBase = datetime.datetime.now().strftime('%Y-%m-%d')
            ctoday = re.sub("今天", dayBase, rowSplit[i])
            rowSplit[i] = ctoday
    rowTime = [0 for i in range(0,len(rowSplit))]
    for i in range(0,len(rowSplit)):
        try:
            rowTime[i] = time.mktime(time.strptime(rowSplit[i], "%Y年%m月%d日 %H:%M"))/60
        except ValueError:
            try:
                rowTime[i] = time.mktime(time.strptime(rowSplit[i], "%Y-%m-%d %H:%M"))/60
            except:
                rowTime[i] = time.mktime(time.strptime(rowSplit[i], "%Y-%m-%d %H:%M:%S"))/60
    rowTime.sort()
    if len(rowTime)-1 < 1:
        weibotimestd = 10000
    else:
        timeInterval = [0 for i in range(0,len(rowTime)-1)]
        for i in range(0,len(rowTime)-1):
            timeInterval[i] = rowTime[i+1]-rowTime[i]
        weibotimestd = np.std(timeInterval,ddof=1)
    print('数据处理完成……')
    return User_Info_Proc(uid = iptuid, fansnum = iptfansnum, follownum = iptfollownum, weibonum = iptweibonum, forwardnum = iptforwardnum, weibotimestd = weibotimestd, totalDivRegtime = totalDivRegtime, forwardDivTotal = forwardDivTotal, registertime = register2now)    

def test(uid):
    try:
        print('接收用户Uid，正在处理……')
        info_raw = proc(get_info(uid))
        sess = tf.InteractiveSession()
        saver=tf.train.import_meta_graph('./model/weibo_modle.meta')#读取网络模型
        saver.restore(sess,'./model/weibo_modle')#读取模型里的各种参数
        graph=tf.get_default_graph()
        X = graph.get_tensor_by_name("input:0")#之前对变量命名在这可以用其名字调出
        Y = graph.get_tensor_by_name("output:0")#占位符一定要全部命名，因为测试时feed_dict需要用到所有的占位符
        print('模型读取完毕！') 
        iptinfo = [info_raw.fansnum, info_raw.follownum, info_raw.forwardnum, info_raw.weibonum, info_raw.weibotimestd, info_raw.totalDivRegtime, info_raw.forwardDivTotal, info_raw.registertime]
        iptinfo = change_data(r'.\userinfo.csv', iptinfo)
        info=np.zeros((1,8))
        for i in range(8):
            info[0][i] = iptinfo[i]
        deres = np.zeros((1,3))#!!
        feed_dict={X:info, Y:deres}
        pred = tf.get_collection('result')[0]
        res = sess.run(pred, feed_dict = feed_dict)
        res = np.eye(res.shape[1])[res.argmax(1)]
        #print(res)
        len = deres.shape[1]
        classes = ['正常用户（0）', '无害僵尸粉（1）','有害僵尸粉（2）']
        for i in range(len):
            if res[0][i] > 0.5:
                print('Uid为'+uid+'的用户是：'+str(classes[i]))
                break
            
    except Exception as e:
        print(e)
        return(0)
 
ipt_uid = input('请输入待检测用户Uid: ')        
test(ipt_uid)