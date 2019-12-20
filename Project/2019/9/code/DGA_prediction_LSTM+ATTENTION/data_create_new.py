# -*- coding:utf-8 -*-
import sys
import codecs
import random

# white 有标签
# balck 没标签
# n 是每类DGA样本的样本个数


def Binary(whiteFilePath, blackDir, trainFilePath, testFilePath, testLabelFilePath,testClassFilePath, leaveOutTestFilePath, leaveOutTestClassFilePath):
	leave_out_classes = ['bedep','beebone','corebot','cryptowall','dircrypt','fobber','hesperbot','matsnu','symmi','tempedreve']

	blacks = {}
	whites = []
	feed_id = {}

	train_file = open(trainFilePath, 'w+')
	test_file = open(testFilePath, 'w+')
	test_label_file = open(testLabelFilePath, 'w+')
	test_classes_file = open(testClassFilePath, 'w+')
	leave_out_file = open(leaveOutTestFilePath, 'w+')
	leave_out_classes_file = open(leaveOutTestClassFilePath, 'w+')


	#读取白样本
	white_file = open(whiteFilePath, 'r')
	lines = white_file.readlines()
	for line in lines:
		whites.append(line.strip('\n'))

	#读取黑样本
	feedFile = open(blackDir + '/feeds.txt', 'r')
	feed_lines = feedFile.readlines()


	#提取全部样例
	i = 1
	for feed_line in feed_lines:
		feed = feed_line.split(' ')[0]
		feed_id[feed] = i
		i += 1

		num = int(feed_line.strip('\n').strip('\r').strip(' ').split(' ')[1])
		#获取全部域名
		domain_list = []
		path = blackDir + '/domains/' + feed + '.txt'
		f = open(path, 'r')
		domain_lines = f.readlines()
		for domain_line in domain_lines:
			domain = domain_line.strip('\n').strip('\r').strip(' ')
			domain_list.append(domain + ' ' + str(i - 1))

		blacks[feed] = domain_list


	train_whites = []
	test_whites = []
	train_blacks = []
	test_blacks = []
	leave_out_blacks = []

	print ('train_whites')

	train_whites = random.sample(whites, int(len(whites) * 4 / 5))
	for index in whites:
		if index not in train_whites:
			test_whites.append(index)
	print ('train_whites finish')

	print ('train_black')
	i = 0
	feed_list = blacks.keys()
	for feed in feed_list:
		if feed in leave_out_classes:
			feed_domains = blacks[feed]
			for index in feed_domains:
				leave_out_blacks.append(index)
		else:
			#带标签
			print (i)
			i += 1
			feed_domains = blacks[feed]
			temp = random.sample(feed_domains, int(len(feed_domains) * 4 / 5))

			for index in feed_domains:
				if index not in temp:
					test_blacks.append(index)
				else:
					train_blacks.append(index)

	print ('train_black finish')



	train_data = [] 	#标签只有0,1
	test_data = []		#标签有0~60
	for index in train_whites:
		train_data.append(index)
	for index in train_blacks:
		train_data.append(index.split(' ')[0] + ' ' + '1')

	for index in test_whites:
		test_data.append(index)
	for index in test_blacks:
		test_data.append(index)

	random.shuffle(train_data)
	random.shuffle(test_data)

	for index in train_data:
		train_file.write(index + '\n')

	for index in test_data:
		test_file.write(index.split(' ')[0] + '\n')
		label = index.split(' ')[1]
		test_classes_file.write(label + '\n')
		if int(label) == 0:
			test_label_file.write(label + '\n')
		else:
			test_label_file.write('1' + '\n')

	for index in leave_out_blacks:
		leave_out_file.write(index.split(' ')[0] + '\n')
		leave_out_classes_file.write(index.split(' ')[1] + '\n')



def Multiclass(n, whiteFilePath, blackDir, trainFilePath, testFilePath, testLabelFilePath):
	blacks = {}
	whites = []

	# 读取白样本
	white_file = open(whiteFilePath, 'r')
	lines = white_file.readlines()
	white_domains = []
	for line in lines:
		white_domains.append(line.strip('\n'))

	whites = random.sample(white_domains, n)

	# 读取黑样本
	# 从所有样例数大于num的种子中随机抽取num个数据组成黑样本

	feedFile = open(blackDir + '/feeds.txt', 'r')
	feed_lines = feedFile.readlines()

	#按要求个数提取样例
	i = 1
	for feed_line in feed_lines:
		feed = feed_line.split(' ')[0]
		num = int(feed_line.strip('\n').strip('\r').strip(' ').split(' ')[1])
		#满足要求的种子
		if num >= n:
			#获取全部域名
			domain_list = []
			path = blackDir + '/domains/' + feed + '.txt'
			f = open(path, 'r')
			domain_lines = f.readlines()
			for domain_line in domain_lines:
				domain = domain_line.strip('\n').strip('\r').strip(' ')
				domain_list.append(domain + ' ' + str(i))

			#从全部域名中随机抽n个
			blacks[feed] = random.sample(domain_list, n)
			i += 1

	print ('sum classes : ' + str(i))


	train_whites = []
	test_whites = []
	train_blacks = []
	test_blacks = []

	print ('train_whites')

	train_whites = random.sample(whites, int(n * 4 / 5))
	for index in whites:
		if index not in train_whites:
			test_whites.append(index)
	print ('train_whites finish')

	print ('train_black')

	feed_list = blacks.keys()
	print (len(feed_list))
	for feed in feed_list:
		feed_domains = blacks[feed]
		temp = random.sample(feed_domains, int(n * 4 / 5))

		for index in feed_domains:
			if index not in temp:
				test_blacks.append(index)
			else:
				train_blacks.append(index)

	print ('train_black finish')

	train_file = open(trainFilePath, 'w+')
	test_file = open(testFilePath, 'w+')
	test_label_file = open(testLabelFilePath, 'w+')

	train_data = []
	test_data = []
	for index in train_whites:
		train_data.append(index)
	for index in train_blacks:
		train_data.append(index)

	for index in test_whites:
		test_data.append(index)
	for index in test_blacks:
		test_data.append(index)

	random.shuffle(train_data)
	random.shuffle(test_data)

	for index in train_data:
		train_file.write(index + '\n')

	for index in test_data:
		test_file.write(index.split(' ')[0] + '\n')
		test_label_file.write(index.split(' ')[1] + '\n')


# Binary('/home/audr/chc/data/white/white.txt',
# 	   '/home/audr/chc/data/black',
# 	   '/home/audr/chc/data/Binary/11.22/train_11.22.txt',
# 	   '/home/audr/chc/data/Binary/11.22/test_11.22.txt',
# 	   '/home/audr/chc/data/Binary/11.22/test_label_11.22.txt',
# 	   '/home/audr/chc/data/Binary/11.22/test_classes_11.22.txt',
# 	   '/home/audr/chc/data/Binary/11.22/leave_out_11.22.txt',
# 	   '/home/audr/chc/data/Binary/11.22/leave_out_classes_11.22.txt'
# 	   )

Multiclass(10000, '/home/audr/chc/data/white/white.txt',
	   '/home/audr/chc/data/black',
	   '/home/audr/chc/data/Multiclass/11.22/train_11.22_10000.txt',
	   '/home/audr/chc/data/Multiclass/11.22/test_11.22_10000.txt',
	   '/home/audr/chc/data/Multiclass/11.22/test_label_11.22_10000.txt')
