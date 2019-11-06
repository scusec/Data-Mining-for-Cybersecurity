import os
import sys
import re
import matplotlib
import pandas as pd
import numpy as np
from os.path import splitext
import tldextract
import whois
import datetime


featureSet = pd.DataFrame(columns=('sql','num_f','capital_f','key_num','space_f','special_f','prefix_f','label'))

PERCENTAGE=1


def get_numf(x):
    length=len(re.compile(r'\d').findall(x))
    return length/len(x)

def get_capitalf(x):
    length=len(re.compile(r'[A-Z]').findall(x))
    return length/len(x) 

def get_keynum(x):
    x=x.lower()
    key_num=x.count('and%20')+x.count('or%20')+x.count('xor%20')+x.count('sysobjects%20')+x.count('version%20')+x.count('substr%20')+x.count('len%20')+x.count('substring%20')+x.count('exists%20')
    key_num=key_num+x.count('mid%20')+x.count('asc%20')+x.count('inner join%20')+x.count('xp_cmdshell%20')+x.count('version%20')+x.count('exec%20')+x.count('having%20')+x.count('unnion%20')+x.count('order%20')+x.count('information schema')
    key_num=key_num+x.count('load_file%20')+x.count('load data infile%20')+x.count('into outfile%20')+x.count('into dumpfile%20')+x.count('select')+x.count('SELECT')
    return key_num


def get_spacef(x):
    if len(x)!=0:
        space_f=(x.count(" ")+x.count("%20"))/len(x)
    return space_f


def get_specialf(x):
    if len(x)!=0:    
        special_f=(x.count("{")*2+x.count('28%')*2+x.count('NULL')+x.count('[')+x.count('=')+x.count('?'))/len(x)
    return special_f

def get_prefixf(x):
    if len(x)!=0:
        prefix_f=(x.count('\\x')+x.count('&')+x.count('\\u')+x.count('%'))/len(x) 
    return prefix_f
    
    


    
    



def getFeatures(sql, label): 
    result = []
    sql=str(sql)
    result.append(sql)
    result.append(get_numf(sql))
    result.append(get_capitalf(sql))
    result.append(get_keynum(sql))
    result.append(get_spacef(sql))
    result.append(get_specialf(sql))
    result.append(get_prefixf(sql))
    result.append(str(label))
    return result



def main():
    data_1=pd.read_csv("normal.csv")
    data_1["label"]=0
    data_2=pd.read_csv("sql.csv")
    data_2["label"]=1
    data=data_1.append(data_2)
    data= data.sample(frac=1).reset_index(drop=True)
    len_of_data=len(data)
    length=int(len_of_data*PERCENTAGE)
    for i in range(length):
        features = getFeatures(data['sql'].loc[i], data["label"].loc[i])    
        featureSet.loc[i] = features


if __name__ == "__main__":
    main()
    featureSet.to_csv('feature.csv',index=None)
    print(featureSet['label'].value_counts())
