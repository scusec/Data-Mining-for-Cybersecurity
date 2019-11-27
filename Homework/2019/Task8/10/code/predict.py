#!/usr/bin/env python
# coding: utf-8


#load模型,并对爬取的图片预测，判断准确率


import pickle
import requests
from matplotlib import pyplot as plt 
from PIL import Image
import numpy as np
from keras.preprocessing import image


from keras.models import load_model

model = load_model("output/captcha__model.h5")


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


url1='http://127.0.0.1:8080/Kaptcha.jpg'
url2='http://127.0.0.1:8080/test.jsp'


s=requests.session()

temp = np.zeros( (height, width, 3),dtype = np.uint8)

#总共预测数量
all_code = 100
#预测正确数量
right_code =0

#预测结果比例
all_results=[]

for j in range(20):
	right_code =0
	for i in range(0,all_code):
		r1 = s.get(url1)
		r2 = s.get(url2)

		code = r2.text.replace('\r\n','')
		code = code.split('_')[0]
		with open('test.jpg', 'wb') as f:
			for chunk in r1.iter_content(chunk_size=1024):
				if chunk:  # filter out keep-alive new chunks
					f.write(chunk)
					f.flush()
			f.close()
		
		
		
		X = np.zeros((1, height, width, 3), dtype = np.uint8)
		img_path = 'test.jpg'
		#读取图片
		raw_img = image.load_img(img_path, target_size=(height, width))
		#讲图片转为np数组
		X[0] = image.img_to_array(raw_img)
		
		result = model.predict(X)
		vex_test = vec_to_captcha(result[0])
		print("真实值："+code+"预测值："+vex_test)
		if code == vex_test:
			right_code=right_code+1
	results = "准确率："+str(right_code/all_code)+"\n"
	print(results)
	all_results.append(results)

for i in range(len(all_results)):
	print(all_results[i])





