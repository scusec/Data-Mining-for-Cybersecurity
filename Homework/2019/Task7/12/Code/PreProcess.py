# coding=utf-8
import os,sys
import codecs
import re
print(sys.path[0])
damahtmldir = "../data//eduhtml"
list = os.listdir(damahtmldir)
dealdata = open("edudealdata",'w')
dealdata.truncate()
dealdata = open("edudealdata",'a')
def calTagNum(path):
    f = codecs.open(path,'r',"utf-8")
    tagnum = str(int((f.read()).count("<")/2))
    f.close()
    return tagnum

def callines(path):
    count = 0
    f = codecs.open(path, 'r', "utf-8")
    for line in f:
        count= count+1
    return count

def calTagType(path):
    pattern_str = r'<([!a-zA-Z0-9]{1,16}?)[ >]'
    pattern = re.compile(pattern_str)

    f = codecs.open(path, 'r', "utf-8")
    dic = []
    for line in f:
        x = pattern.findall(line)
        dic = dic + x

    tagtype =  (len(set(dic)))
    f.close()
    return tagtype

for file in list:
    path = os.path.join(damahtmldir, file)
    if(os.path.isfile(path)):

        fsize = os.path.getsize(path)
        ftagnum = calTagNum(path)
        flinecount = callines(path)
        ftagtype = calTagType(path)
        fflag = 0

        dealdata.write(str(fsize)+'\t'+str(ftagnum)+'\t'+str(flinecount)+'\t'+str(ftagtype)+'\t'+str(fflag)+'\n')

dealdata.close()