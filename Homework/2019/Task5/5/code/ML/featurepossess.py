# -*- coding: UTF-8 -*-
import re

def generate(odir,wdir,label):
    f_input=open(wdir, 'w')
    with open(odir, 'rb') as f:
        data = [x.decode('utf-8').strip() for x in f.readlines()]
        print(data)
        line_number=0

        for line in data:
            global feature
            num_len=0
            capital_len=0
            key_num=0
            feature3=0
            line_number=line_number+1
            num_len=len(re.compile(r'\d').findall(line))
            if len(line)!=0:
                num_f=num_len/len(line)#数字字符频率
            capital_len=len(re.compile(r'[A-Z]').findall(line))
            if len(line)!=0:
                capital_f=capital_len/len(line)#大写字母频率
            line=line.lower()

            key_num=line.count('and%20')+line.count('or%20')+line.count('xor%20')+line.count('sysobjects%20')+line.count('version%20')+line.count('substr%20')+line.count('len%20')+line.count('substring%20')+line.count('exists%20')
            key_num=key_num+line.count('mid%20')+line.count('asc%20')+line.count('inner join%20')+line.count('xp_cmdshell%20')+line.count('version%20')+line.count('exec%20')+line.count('having%20')+line.count('union%20')+line.count('order%20')+line.count('information schema')
            key_num=key_num+line.count('load_file%20')+line.count('load data infile%20')+line.count('into outfile%20')+line.count('into dumpfile%20')
            if len(line)!=0:
                space_f=(line.count(" ")+line.count("%20"))/len(line)#空格百分比
                special_f=(line.count("{")*2+line.count('28%')*2+line.count('NULL')+line.count('[')+line.count('=')+line.count('?'))/len(line)
                comments_f =(line.count("#")+line.count('--+'))
                prefix_f=(line.count('\\x')+line.count('&')+line.count('\\u')+line.count('%'))/len(line)
            #print('%f,%f,%f,%f,%f,%f,%f,%f' % (len(line),key_num,capital_f,num_f,space_f,special_f,prefix_f,label))

            #找不完整的引号数量


            f_input.write('%f,%f,%f,%f,%f,%f,%f,%f,%f' % (len(line),key_num,capital_f,num_f,space_f,special_f,prefix_f,comments_f,label)+'\n')

    f_input.close()
    return wdir
