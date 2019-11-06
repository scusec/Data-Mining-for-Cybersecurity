with open("2.txt",'r') as f1,open("3.txt",'w') as f2:
    lines=f1.readlines()
    lines=set(lines)
    for line in lines:
        f2.writelines(line)