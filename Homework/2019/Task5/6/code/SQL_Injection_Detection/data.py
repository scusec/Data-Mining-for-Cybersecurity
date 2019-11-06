import feature
import numpy as np
import pandas as pd
def dataGenerate(soursefile,targetfile,label):
    fileInput = open(targetfile,'w')
    with open(soursefile,'rb') as f:
        data = [x.decode('utf-8').strip() for x in f.readlines()]
        for line in data:
            if len(line) != 0:
                entropy = feature.getUrlEntropy(line)
            
            if len(line) != 0:
                length = feature.getLength(line)
            
            if len(line) != 0:
                num = feature.getNum(line)
            
            if len(line) != 0:
                capitalPro = feature.getProCapital(line)
            
            if len(line) != 0:
                evilWord = feature.getEvilWord(line)
            
            if len(line) != 0:
                maxLength = feature.getMaxLength(line)
            
            fileInput.write('%f,%f,%f,%f,%f,%f,%f' %(entropy,length,num,capitalPro,evilWord,maxLength,label)+'\n')
    
    fileInput.close()
    return targetfile

sqlMatrix = dataGenerate("./Data/sqlnew.csv","./Data/sqlMatrix.csv",0)
normalMatrix = dataGenerate("./Data/normal_less.csv","./Data/normalMatrix.csv",1)

df = pd.read_csv(sqlMatrix)
df.to_csv("./Data/all_data.csv",encoding="utf_8_sig",index=False)
df = pd.read_csv(normalMatrix)
df.to_csv("./Data/all_data.csv",encoding="utf_8_sig",index=False,header=False,mode='a+')
print("特征数据集存储成功")