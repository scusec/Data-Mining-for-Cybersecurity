fin = open('./train_11.22_10000.txt', 'r')
fout1 = open('./train_test_11.22_10000.txt', 'w')
fout2 = open('./train_test_label_11.22_10000.txt', 'w')
lines = fin.readlines()

for line in lines:
    domains = line.strip(' ').strip('\n').split(' ')[0]
    label = line.strip(' ').strip('\n').split(' ')[1]
    fout1.write(domains + '\n')
    fout2.write(label + '\n')