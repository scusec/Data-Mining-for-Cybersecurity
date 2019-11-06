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