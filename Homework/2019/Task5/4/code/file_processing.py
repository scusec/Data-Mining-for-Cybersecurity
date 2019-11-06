import os
# 注意文件中不能出现换行符占一行，不然会提前终端

def return_files(rootDir):
    list_dirs = os.walk(rootDir)
    funfiles=[]
    for root, dirs, files in list_dirs:
        for f in files:
            funfiles.append(os.path.join(root, f))
    return funfiles

def put_in_one(attack_type):
    fwrite= open(attack_type+'.txt','w') 
    fwrite.close() # 刷新文件内容

    fwrite= open(attack_type+'.txt','a',encoding='utf8') 
    files=return_files(attack_type)
    for vfile in files:
        print('write the file:'+vfile)
        with open(vfile,encoding='utf8') as fread:
            lines=fread.readlines()
            fwrite.writelines(lines)
            fwrite.write('\n')
    fwrite.close()



# put_in_one('data/codeinjection')
put_in_one('data/directorytraversal')
put_in_one('data/sqli')
put_in_one('data/xss')