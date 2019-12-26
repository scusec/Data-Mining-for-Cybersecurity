from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
import re
import os
import csv

rootdir1 = "E:\\normal_apk"
rootdir2 = "E:\\malicious_apk"

def getPermissions(apkname):
    try:
        app = apk.APK(apkname)
        permissions = app.get_permissions()
        permissions_list = []
        for i in permissions:
            tem = i.split(".")
            result = tem[-1]
            permissions_list.append(result)
        print(permissions_list)
        return permissions_list
    except Exception:
        pass


if __name__ == "__main__":
    '''
    file = open("./data/malicious_permissions.csv", "a", newline="")
    writer = csv.writer(file)
    count = 0
    for dirpath, dirnames, filenames in os.walk(rootdir1):
        for filename in filenames:
            permissions = getPermissions(rootdir1 + "\\" + filename)
            if permissions:
                permissions.insert(0, 0)
                writer.writerow(permissions)
                print(count)
                count = count + 1
    file.close()
    '''

    file = open("./data/normal_permissions.csv", "a", newline="")
    writer = csv.writer(file)
    count = 0
    for dirpath, dirnames, filenames in os.walk(rootdir2):
        for filename in filenames:
            permissions = getPermissions(rootdir2 + "\\" + filename)
            if permissions:
                permissions.insert(0, 1)
                writer.writerow(permissions)
                print(count)
                count = count + 1
    file.close()







