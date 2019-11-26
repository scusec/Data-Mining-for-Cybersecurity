import pickle
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
from matplotlib import pyplot as plt

#验证码所包含的字符
captcha_word = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

#图片的长度和宽度
width = 160
height = 60

#每个验证码所包含的字符数
word_len = 4

#字符总数
word_class = len(captcha_word)

#读取pickle文件
file = open('../dataset/captcha_train_data.pkl', 'rb')
X, y = pickle.load(file)

#创建输入，结构为 高，宽，通道
input_tensor = Input(shape=(height, width, 3))

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
x = Convolution2D(128, 3, padding='same',activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
x = Flatten()(x)
#为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，Dropout层用于防止过拟合。
x = Dropout(0.25)(x)

#Dense就是常用的全连接层
#最后连接5个分类器，每个分类器是word_len个神经元，分别输出各自字符的概率。
x = [Dense(word_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(word_len)]
output = concatenate(x)

#构建模型
model = Model(inputs=input_tensor, outputs=output)

#因为训练可能需要数个小时，所以这里加载了之前训练好的参数。准确率97%左右
model.load_weights('../output/weights.16(0.9662+0.9975).hdf5')

#这里优化器选用Adadelta，学习率0.1
opt = Adadelta(lr=0.1)
#编译模型以供训练，损失函数使用 categorical_crossentropy，使用accuracy评估模型在训练和测试时的性能的指标
model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#每次epoch都保存一下权重，用于继续训练
checkpointer = ModelCheckpoint(filepath="../output/weights.{epoch:02d}.hdf5", monitor='val_acc',
                               verbose=2, save_weights_only=True)

print('Model train start.')
#开始训练，validation_split代表10%的数据不参与训练，用于做验证集
model.fit(X, y, epochs= 20, callbacks=[checkpointer], validation_split=0.1)
print('Model train done.')

#保存权重和模型
model.save_weights('../model/captcha_model_weights.h5')
model.save('../model/captcha__model.h5')