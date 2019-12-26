#打草稿用的文件
import csv
from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
from hashlib import md5

'''
permissionsList = []
classVec = []
file = open("./data/malicious_permissions.csv", "r", newline="")
read = csv.reader(file)
for row in read:
    classVec.append(int(row[0]))
    del row[0]
    permissionsList.append(row)
'''

app = apk.APK("./sample/玉米直播.apk")
print(app.get_signature_name())

m = md5()
file = open("./sample/玉米直播.apk", "rb")
m.update(file.read())
app_md5 = m.hexdigest()
file.close()
print(app_md5)


