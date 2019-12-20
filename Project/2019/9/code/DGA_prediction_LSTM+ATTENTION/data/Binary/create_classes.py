# -*- coding:utf-8 -*-

feed_id = {}
feed_domains = {}

test_domains = []
test_domains_label = []
test_domains_classes = []



feedFile = open('./../black/feeds.txt', 'r')
lines = feedFile.readlines()
i = 1
for line in lines:
    feed_id[line.split(' ')[0]] = i
    feed_domains[line.split(' ')[0]] = []
    i += 1
feedFile.close()

f = open('./test_11.18.txt', 'r')
lines = f.readlines()
for line in lines:
    test_domains.append(line.strip('\n').strip('\r').strip(' '))
f.close()

f = open('./test_label_11.18.txt', 'r')
lines = f.readlines()
for line in lines:
    test_domains_label.append(int(line.strip('\n').strip('\r').strip(' ')))
f.close()



feed_list = feed_id.keys()

for feed in feed_list:
    path = './../black/domains/' + feed + '.txt'
    f = open(path , 'r')
    lines = f.readlines()
    for line in lines:
        feed_domains[feed].append(line.strip('\n').strip('\r').strip(' '))
    f.close()

for i in range(len(test_domains)):
    print i
    if test_domains_label[i] == 0 :
        test_domains_classes.append(0)
    else:
        for feed in feed_list:
            if test_domains[i] in feed_domains[feed]:
                test_domains_classes.append(feed_id[feed])
                break

f = open('./test_classes_11.18.txt', 'w')
for i in test_domains_classes:
    f.write(str(i) + '\n')