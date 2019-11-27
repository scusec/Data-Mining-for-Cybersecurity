
# coding: utf-8

#爬取图片，并生成'captcha_train_data.pkl'



import requests
from PIL import Image
import re
import pickle
import numpy as np
import os
from matplotlib import pyplot as plt 
from PIL import Image
import shutil
from keras.preprocessing import image


#包含字符 
captcha_word = "_0123456789abcdefghijklmnopqrstuvwxyz"
#图片的长度和宽度
width = 180
height = 60
#字符数
word_len = 4
#总数
word_class = len(captcha_word)
#验证码素材目录
dataset_dir = 'dataset'

#生成字符索引，同时反向操作一次，方面还原
char_indices = dict((c, i) for i,c in enumerate(captcha_word))
indices_char = dict((i, c) for i,c in enumerate(captcha_word))

#验证码图片链接
url1='http://127.0.0.1:8080/Kaptcha.jpg'
#验证码字符串链接
url2='http://127.0.0.1:8080/test.jsp'


#验证码字符串转数组
def captcha_to_vec(captcha):    
	#创建一个长度为 字符个数 * 字符种数 长度的数组
	vector = np.zeros(word_len * word_class)
	
	#文字转数组
	for i,ch in enumerate(captcha):
		idex = i * word_class + char_indices[ch]
		vector[idex] = 1
	return vector


#数组转换文字
def vec_to_captcha(vec):
	text = []
	vec = np.reshape(vec,(word_len,word_class))
	
	for i in range(len(vec)):
		temp = vec[i]
		max_index = np.argmax(temp)# 最大值的索引
	
		text.append(captcha_word[max_index % word_class])
		#print(text)
	return ''.join(text)


#爬取图片
def get_jpg(number):
	print("开始爬取图片\n");
	s=requests.session()

	for i in range(0,number):
		r1 = s.get(url1)
		r2 = s.get(url2)

		code = r2.text.replace('\r\n','')
		code = code.replace('\n','')
		code = code.replace('\r','')
		
		with open("dataset/"+code+'.jpg', 'wb') as f:
			for chunk in r1.iter_content(chunk_size=1024):
				if chunk:  # filter out keep-alive new chunks
					f.write(chunk)
					f.flush()
			f.close()
	print("爬取图片结束\n");

#使用图片生成'captcha_train_data.pkl'
def process_dataset():

	#获取目录下样本列表
	image_list = []
	for item in os.listdir(dataset_dir):
		image_list.append(item)
	np.random.shuffle(image_list)

	X = np.zeros((len(image_list), height, width, 3), dtype = np.uint8)
	# 创建数组，储存标签信息
	y = np.zeros((len(image_list), word_len * word_class), dtype = np.uint8 )

	for i,img in enumerate(image_list):
		if i % 100 == 0:
			print(i)
		img_path = dataset_dir + "/" + img
		#读取图片
		raw_img = image.load_img(img_path, target_size=(height, width))
		#讲图片转为np数组
		X[i] = image.img_to_array(raw_img)
		#讲标签转换为数组进行保存
		y[i] = captcha_to_vec(img.split('_')[0])

	file = open('captcha_train_data.pkl','wb')
	pickle.dump((X,y) , file)
	file.close()
	
	#删除文件夹及文件
	shutil.rmtree(dataset_dir)
	#创建文件夹
	os.mkdir(dataset_dir)

if __name__ == "__main__":
	get_jpg(5000)
	process_dataset()


