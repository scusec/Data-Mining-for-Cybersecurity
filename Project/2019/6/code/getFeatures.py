import csv
from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
from hashlib import md5
import os
import math

rootdir1 = "E:\\normal_apk"
rootdir2 = "E:\\malicious_apk"
m = md5()

def getSize(apkname):
    return os.path.getsize(apkname)


def getMd5(apkname):
    file = open(apkname, "rb")
    m.update(file.read())
    app_md5 = m.hexdigest()
    file.close()
    return app_md5


def getSignature(apkname):
    app = apk.APK(apkname)
    return app.get_signature_name()


def getfileComEntropy(apkname):
    word = {}
    p = 0
    sum = 0
    with open(apkname, 'rb') as f:
        content = f.readlines()
        for i in content:
            for j in i:
                if j != '\n' and j != ' ':
                    if j not in word.keys():
                        word[j] = 1
                    else:
                        word[j] = word[j] + 1
                else:
                    pass
    for i in word.keys():
        sum = sum + word[i]
    for i in word.keys():
        p = p - float(word[i])/sum * math.log(float(word[i])/sum, 2)
    return p


if __name__ == "__main__":
   file = open("./data/features.csv", "a", newline="")
   writer = csv.writer(file)
   writer.writerow(['label', 'size', 'md5', 'signature', 'comentropy'])
   count = 0
   for dirpath, dirnames, filenames in os.walk(rootdir1):
       for filename in filenames:
           size = getSize(rootdir1 + "\\" + filename)
           md5 = getMd5(rootdir1 + "\\" + filename)
           signature = getSignature(rootdir1 + "\\" + filename)
           comentropy  =getfileComEntropy(rootdir1 + "\\" + filename)
           writer.writerow(['1', str(size), md5, signature, str(comentropy)])
           count = count + 1
   file.close()

'''
   file = open("./data/features.csv", "a", newline="")
   writer = csv.writer(file)
   count = 0
   for dirpath, dirnames, filenames in os.walk(rootdir2):
       for filename in filenames:
           size = getSize(rootdir2 + "\\" + filename)
           md5 = getMd5(rootdir2 + "\\" + filename)
           signature = getSignature(rootdir2 + "\\" + filename)
           comentropy = getfileComEntropy(rootdir2 + "\\" + filename)
           writer.writerow(['1', str(size), md5, signature, str(comentropy)])
           count = count + 1
   file.close()
'''