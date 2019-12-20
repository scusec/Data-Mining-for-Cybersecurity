# -*- coding:utf-8 -*-
import os
import sys
import Binary
import Binary_with_attention
import Multiclass_with_attention
import Multiclass
import codecs
import numpy as np


if __name__ == '__main__':
	mode = sys.argv[1]		#模式： 0->二分类训练; 1->二分类判别; 2->多分类训练; 3->多分类判别;
	batch_size = int(sys.argv[2])
	epochs = int(sys.argv[3])
	srcFilePath = sys.argv[4]	#原始数据文件路径
	modelPath = sys.argv[5]		#模型文件保存路径或读取路径

	resultFilePath = ''
	nb_classes = 2
	if mode == '1' or mode == '3':
		resultFilePath = sys.argv[6] 	#模型判别结果文件路径
	elif mode == '2':
		nb_classes = int(sys.argv[6])	#多分类训练时数据集中数据的种类

	#读取配置文件
	charList = {}
	confFilePath = sys.path[0] + '/configFiles/charList.txt'
	confFile = codecs.open(filename=confFilePath, mode='r', encoding='utf-8', errors='ignore')
	lines = confFile.readlines()
	#字符序列要从1开始,0是填充字符
	i = 1
	for line in lines:
		temp = line.strip('\n').strip('\r').strip(' ')
		if temp != '':
			charList[temp] = i
			i += 1


	max_features = i

	#设置批处理个数与训练轮数
	# Binary_batch_size = 180
	# Binary_epochs = 8
	# Multiclass_batch_size = 100
	# Multiclass_epochs = 15

	#转换数据格式
	x_data_sum = []
	y_data_sum = []

	if mode == '0':
		srcFile = codecs.open(filename=srcFilePath, mode='r', encoding='utf-8', errors='ignore')
		lines = srcFile.readlines()
		for line in lines:
			if line.strip('\n').strip('\r').strip(' ') == '':
				continue

			x_data = []
			s = line.strip('\n').strip('\r').strip(' ').split(' ')
			x = str(s[0])
			y = int(s[1])

			for char in x :
				try:
					x_data.append(charList[char])
				except:
					print ('unexpected char' + ' : '+ char)
					x_data.append(0)

			x_data_sum.append(x_data)
			y_data_sum.append(y)

		x_data_sum = np.array(x_data_sum)
		y_data_sum = np.array(y_data_sum)

		# Binary.train(max_features, x_data_sum, y_data_sum, batch_size, epochs, modelPath)
		Binary_with_attention.train(max_features, x_data_sum, y_data_sum, batch_size, epochs, modelPath)



	elif mode == '1':
		srcFile = codecs.open(filename=srcFilePath, mode='r', encoding='utf-8', errors='ignore')
		lines = srcFile.readlines()
		for line in lines:
			if line.strip('\n').strip('\r').strip(' ') == '':
				continue

			x_data = []
			x = line.strip('\n').strip('\r').strip(' ')

			for char in x:
				try:
					x_data.append(charList[char])
				except:
					print ('unexpected char' + ' : ' + char)
					x_data.append(0)

			x_data_sum.append(x_data)

		x_data_sum = np.array(x_data_sum)

		# Binary.predict(x_data_sum, batch_size, modelPath, resultFilePath)
		Binary_with_attention.predict(x_data_sum, batch_size, modelPath, resultFilePath)

	elif mode == '2':
		srcFile = codecs.open(filename=srcFilePath, mode='r', encoding='utf-8', errors='ignore')
		lines = srcFile.readlines()
		for line in lines:
			if line.strip('\n').strip('\r').strip(' ') == '':
				continue

			x_data = []
			y_data = []
			for i in range(nb_classes):
				y_data.append(0)

			s = line.strip('\n').strip('\r').strip(' ').split(' ')
			x = str(s[0])
			y = int(s[1])
			y_data[y] = 1
			for char in x:
				try:
					x_data.append(charList[char])
				except:
					print ('unexpected char' + ' : ' + char)
					x_data.append(0)

			x_data_sum.append(x_data)
			y_data_sum.append(y_data)

		x_data_sum = np.array(x_data_sum)
		y_data_sum = np.array(y_data_sum)

		# Multiclass.train(max_features, nb_classes, x_data_sum, y_data_sum, batch_size, epochs, modelPath)
		Multiclass_with_attention.train(max_features, nb_classes, x_data_sum, y_data_sum, batch_size, epochs, modelPath)
	elif mode == '3':
		srcFile = codecs.open(filename=srcFilePath, mode='r', encoding='utf-8', errors='ignore')
		lines = srcFile.readlines()
		for line in lines:
			if line.strip('\n').strip('\r').strip(' ') == '':
				continue

			x_data = []
			x = line.strip('\n').strip('\r').strip(' ')

			for char in x:
				try:
					x_data.append(charList[char])
				except:
					print ('unexpected char' + ' : ' + char)
					x_data.append(0)

			x_data_sum.append(x_data)

		x_data_sum = np.array(x_data_sum)

		# Multiclass.predict(x_data_sum, batch_size, modelPath, resultFilePath)
		Multiclass_with_attention.predict(x_data_sum, batch_size, modelPath, resultFilePath)
	else:
		pass