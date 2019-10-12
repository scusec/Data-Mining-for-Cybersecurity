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



featureSet = pd.DataFrame(columns=('payload','dots','\"','\'','java',\
'script','alert','%','<','>','style','iframe','97','108','\x60','&#x28','label'))

PERCENTAGE=1



def countdots(payload):  
    return payload.count('.')

def isPresentDSlash(payload):
    return payload.count('\"')

def countSubDir(payload):
    return payload.count('\'')

def countjava(payload):
    return payload.lower().count('java')

def countscript(payload):
    return payload.lower().count('script')

def countalert(payload):
    return payload.lower().count('alert')

def countpercent(payload):
    return payload.count('%')

def countAngleBrackets_1(payload):
    return payload.count('<')

def countAngleBrackets_2(payload):
    return payload.count('>')

def countstyle(payload):
    return payload.lower().count('style')

def countiframe(payload):
    return payload.lower().count('iframe')

def count97(payload):
    return payload.count('97')
    
def count108(payload):
    return payload.count('108')
    
def count60(payload):
    return payload.count('\x60')
    
def count28(payload):
    return payload.count('&#x28')
    
    



def getFeatures(payload, label): 
    result = []
    payload = str(payload)
    
    #add the payload to feature set
    result.append(payload)
    result.append(countdots(payload))
    result.append(isPresentDSlash(payload))
    result.append(countSubDir(payload))
    result.append(countjava(payload))
    result.append(countscript(payload))
    result.append(countalert(payload))
    result.append(countpercent(payload))
    result.append(countAngleBrackets_1(payload))
    result.append(countAngleBrackets_2(payload))
    result.append(countstyle(payload))
    result.append(countiframe(payload))
    result.append(count97(payload))
    result.append(count108(payload))
    result.append(count60(payload))
    result.append(count28(payload))
    result.append(str(label))
    return result



def main():
    df_1 = pd.read_csv("xssed.csv")
    df_2=pd.read_csv("dmzo_nomal.csv")
    df_2['label']=1
    df=df_1.append(df_2)
    df = df.sample(frac=1).reset_index(drop=True)
    len_of_data=len(df)
    length=int(len_of_data*PERCENTAGE)
    for i in range(length):
        features = getFeatures(df["payload"].loc[i], df["label"].loc[i])    
        featureSet.loc[i] = features


if __name__ == "__main__":
    main()
    # print(featureSet["payload"])
    featureSet.to_csv('feature_1.csv',index=None)
    print(featureSet['label'].value_counts())
