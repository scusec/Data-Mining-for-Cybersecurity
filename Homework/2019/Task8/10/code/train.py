#!/usr/bin/env python
# coding: utf-8

#训练模型

import os
import pickle

from keras.utils import np_utils
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
from matplotlib import pyplot as plt 
from keras.models import load_model

import get_dataset

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

#创建输入，结构为 高，宽，3通道
input_tensor = Input( shape=(height, width, 3))

x = input_tensor

#构建卷积网络
#两层卷积层，一层池化层，重复3次。因为生成的验证码比较小，padding使用same
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)


x = Convolution2D(64, 3, padding='same', activation='relu')(x)
x = Convolution2D(64, 3, padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)


x = Convolution2D(128, 3, padding='same', activation='relu')(x)
x = Convolution2D(128, 3, padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
x = Flatten()(x)
#防止过拟合
x = Dropout(0.25)(x)

#全连接层
x = [Dense(word_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(word_len)]
output = concatenate(x)


if 'captcha__model.h5' in os.listdir("output"):
	print("load模型")
	model = load_model("output/captcha__model.h5")
else:
	opt = Adadelta(lr=0.1)
	model = Model(inputs=input_tensor, outputs=output)
	model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


from keras.utils.vis_utils import plot_model
MODEL_VIS_FILE = 'output/captcha_classfication' + '.png'
# 模型可视化
plot_model(model,to_file=MODEL_VIS_FILE,show_shapes=True)


for i in range(10000):
	if i%5 == 0:
		#爬取5000张图片
		get_dataset.get_jpg(5000)
		#生成'captcha_train_data.pkl'
		get_dataset.process_dataset()
		file = open('captcha_train_data.pkl', 'rb')
		X, y = pickle.load(file)
	#训练
	model.fit(X, y, epochs= 1, validation_split=0.1)
	#保存权重模型
	model.save_weights('output/captcha_model_weights.h5')
	model.save('output/captcha__model.h5')
	



