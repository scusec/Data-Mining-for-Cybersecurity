import pandas as  pd
import numpy as np 
import re
import urllib
import math
from sklearn.utils import shuffle
from urllib import parse as urlparse

#获取URL的信息熵
def getUrlEntropy(url):
    #print("typeurl:",type(url))
    lableCount = {}
    url = str(url)
    length = len(url)
    for i in range(0,length):
        if url[i] in lableCount.keys():
            lableCount[url[i]] = lableCount[url[i]] + 1
        else:
            lableCount[url[i]] = 1
    
    shangnon = 0
    for i in lableCount.keys():
        prob = float(lableCount[i]) / length
        shangnon = shangnon - prob * math.log(prob, 2)
    return shangnon

#获取url的长度
def getLength(url):
    return len(url)

#获取最长参数的长度
def getMaxLength(url):
    parsed = urlparse.urlparse(urllib.parse.unquote(url))
    urlQuery = urlparse.parse_qs(parsed.query,True)
    urlFistArgLength = 0
    if len(urlQuery) == 0:
        urlFistArgLength = 0
    elif len(urlQuery) == 1:
        urlFistArgLength = len(urlQuery[list(urlQuery.keys())[0]][0])
    else:
        maxLen = 0
        for i in urlQuery.keys():
            if len(urlQuery[i][0]) > maxLen:
                maxLen = len(urlQuery[i][0])
        urlFistArgLength = maxLen
    
    return urlFistArgLength


#获取数字字符的频率
def getNum(url):
    num_len = len(re.compile(r'\d').findall(url))
    return float(num_len / len(url))

#获取大写字母频率
def getProCapital(url):
    capital_len = len(re.compile(r'[A-Z]').findall(url))
    return float(capital_len / len(url) )


#统计非法字符的数量
def getEvilWord(url):
    url = url.lower()
    url_Evil_Word=url.count('and%20')+url.count('or%20')+url.count('xor%20')+url.count('sysobjects%20')+url.count('version%20')+url.count('substr%20')+url.count('len%20')+url.count('substring%20')+url.count('exists%20')+url.count('--')+url.count('&&')
    url_Evil_Word=url_Evil_Word+url.count('mid%20')+url.count('asc%20')+url.count('inner join%20')+url.count('xp_cmdshell%20')+url.count('version%20')+url.count('exec%20')+url.count('having%20')+url.count('unnion%20')+url.count('order%20')+url.count('information schema')
    url_Evil_Word=url_Evil_Word+url.count('load_file%20')+url.count('load data infile%20')+url.count('into outfile%20')+url.count('into dumpfile%20')
    return url_Evil_Word


            