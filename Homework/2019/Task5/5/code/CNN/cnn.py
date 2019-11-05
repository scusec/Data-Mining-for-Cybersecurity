
import os  
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import random,pickle,time
from math import ceil
from word import batch_generator,data_generator
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.layers import Dense,InputLayer,Dropout,Convolution1D,Flatten,GlobalAveragePooling1D,MaxPool1D
from keras.utils import plot_model
from IPython import display

NB_EPOCH = 5
BATCH_SIZE = 100
#梯度下降算法
OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
input_shape=0
lens=0
log_dir="./log/"
class CNN:
	@staticmethod
	def build(input_shape, classes):
		model = Sequential()
		# CONV => RELU => POOL
		# 第一个卷积层，16个卷积核，大小3*3，卷积模式SAME,激活函数relu,输入张量的大小
		model.add(Convolution1D(16, kernel_size=3, padding="same",
			input_shape=input_shape,W_regularizer=l2(0.01),b_regularizer=l2(0.01)))
		model.add(Activation("relu"))
		# 池化层,池化核大小２x2
		model.add(MaxPool1D(pool_size=2))
		# CONV => RELU => POOL
		model.add(Convolution1D(32, kernel_size=4, padding="same",
			input_shape=input_shape,W_regularizer=l2(0.01),b_regularizer=l2(0.01)))
		model.add(Activation("relu"))
		model.add(MaxPool1D(pool_size=2))
		# CONV => RELU => POOL
		model.add(Convolution1D(64, kernel_size=5, padding="same",
			input_shape=input_shape,W_regularizer=l2(0.01),b_regularizer=l2(0.01)))
		model.add(Activation("relu"))
		model.add(MaxPool1D(pool_size=2))
		# Flatten => RELU layers
		model.add(Dropout(0.5))
		# 全连接层,展开操作
		model.add(Flatten())
		# 添加隐藏层神经元的数量和激活函数
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
 
		# a softmax classifier
		model.add(Dense(classes))
		# 输出层
		model.add(Activation("softmax"))
		# 模型可视化
		plot_model(model, to_file="SQL_CNN.png", show_shapes=True)
		display.Image('SQL_CNN.png')
		return model

def train_cnn():
	for line in open("./file/INPUT_SHAPE"):
		input_shape=int(line)
		print(input_shape)
	INPUT_SHAPE=(input_shape,16)
	for line in open("./file/len"):
		lens=int(line)
		print(lens)
	data_size=ceil(lens//(BATCH_SIZE*NB_EPOCH))
	print(data_size)
	for line in open("./file/valid_len"):
		valid_lens=int(line)
		print(valid_lens)
	valid_size=ceil(valid_lens//(BATCH_SIZE*NB_EPOCH))
	print(valid_size)
	model = CNN.build(input_shape=INPUT_SHAPE, classes=3)
	print('1')
	model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
		metrics=["accuracy"])
	print('2')
	call=TensorBoard(log_dir=log_dir+"cnn",write_grads=True)
	checkpoint = ModelCheckpoint(filepath='cnn.h5',monitor='val_acc',mode='auto' ,save_best_only='True')
	next_batch=data_generator(BATCH_SIZE,input_shape,"./file/x_train")
	next_valid_batch=data_generator(BATCH_SIZE,input_shape,"./file/x_valid")
	model.fit_generator(batch_generator(next_batch,data_size),steps_per_epoch=data_size,
		epochs=NB_EPOCH,callbacks=[call,checkpoint],validation_data=batch_generator(next_valid_batch,data_size),
		nb_val_samples=valid_size)
	print('3')
	#model.save('cnn.h5')
	print("model save complete!")



train_cnn()