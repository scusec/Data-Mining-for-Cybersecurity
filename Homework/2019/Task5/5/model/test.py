from sklearn.externals import joblib
import re
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def generate(payload):
    num_len=0
    capital_len=0
    key_num=0
    feature3=0
    num_len=len(re.compile(r'\d').findall(payload))
    if len(payload)!=0:
        num_f=num_len/len(payload)#数字字符频率
        capital_len=len(re.compile(r'[A-Z]').findall(payload))
    if len(payload)!=0:
            capital_f=capital_len/len(payload)#大写字母频率
    payload=payload.lower()
    key_num=payload.count('and%20')+payload.count('or%20')+payload.count('xor%20')+payload.count('sysobjects%20')+payload.count('version%20')+payload.count('substr%20')+payload.count('len%20')+payload.count('substring%20')+payload.count('exists%20')
    key_num=key_num+payload.count('mid%20')+payload.count('asc%20')+payload.count('inner join%20')+payload.count('xp_cmdshell%20')+payload.count('version%20')+payload.count('exec%20')+payload.count('having%20')+payload.count('union%20')+payload.count('order%20')+payload.count('information schema')
    key_num=key_num+payload.count('load_file%20')+payload.count('load data infile%20')+payload.count('into outfile%20')+payload.count('into dumpfile%20')
    if len(payload)!=0:
        space_f=(payload.count(" ")+payload.count("%20"))/len(payload)#空格百分比
        special_f=(payload.count("{")*2+payload.count('28%')*2+payload.count('NULL')+payload.count('[')+payload.count('=')+payload.count('?'))/len(payload)
        comments_f =(payload.count("#")+payload.count('--+'))
        prefix_f=(payload.count('\\x')+payload.count('&')+payload.count('\\u')+payload.count('%'))/len(payload)
    matrix = [len(payload),key_num,capital_f,num_f,space_f,special_f,prefix_f,comments_f]
    return matrix

payload =str(input("please input ur payload:"))
model_path = str(input("please input the model you want to test:"))
clf = joblib.load("./file/"+model_path)
payload_vec = np.array(generate(payload)).reshape(1,-1)
res = clf.predict(payload_vec)
#print("%.3f"%res)
if res==0:
    print("不是sql注入吧！")
else:
    print("是sql注入吧！！！")
