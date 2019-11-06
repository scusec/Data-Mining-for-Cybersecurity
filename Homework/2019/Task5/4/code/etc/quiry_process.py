import re

with open('normalTrafficTraining.txt','r') as f,open('1.txt','w')as f2:
    line=' '
    lastLine='\n'
    payhead=''
    while(lastLine !=''):
        try:
            lastLine = line
            line = f.readline()
        except EOFError as e:
            f.close()
            break
        if lastLine=='\n':

            if 'POST' in line:
                payhead = re.findall('POST http://[^/]*/(.*) HTTP', line)[0]
                # print("the payhead is:" + payhead)
            elif 'GET' in line or 'PUT' in line:
                #print(line)
                pay = re.findall('[GET|PUT] http://[^/]*[/|.|~](.*) HTTP', line)[0]
                # print("GET:" + pay)  # 打印匹配到的字符
                f2.writelines(pay+'\n')
            elif line !='\n':
                pay = payhead + '?' + line
                # print("POST:"+pay)
                f2.writelines(pay+'\n')

with open("1.txt",'r') as f1,open("2.txt",'w') as f2:
    line2=''
    for line in f1:
        if(line!='\n'):
            f2.writelines(line)


with open("2.txt",'r') as f1,open("3.txt",'w') as f2:
    lines=f1.readlines()
    lines=set(lines)
    for line in lines:
        f2.writelines(line)