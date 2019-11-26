import os
import numpy as np
import pickle
from keras.preprocessing import image

#验证码所包含的字符
captcha_word = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

#图片的长度和宽度
width = 160
height = 60

#每个验证码所包含的字符数
word_len = 4
#字符总数
word_class = len(captcha_word)

#验证码素材目录
train_dir = '../train'

#生成字符索引，同时反向操作一次，方面还原
char_indices = dict((c, i) for i,c in enumerate(captcha_word))
indices_char = dict((i, c) for i,c in enumerate(captcha_word))

#验证码字符串转数组
def captcha_to_vec(captcha):    
    #创建一个长度为 字符个数 * 字符种数 长度的数组
    vector = np.zeros(word_len * word_class)
    
    #文字转成成数组
    for i,ch in enumerate(captcha):
        idex = i * word_class + char_indices[ch]
        vector[idex] = 1
    return vector

#把数组转换回文字
def vec_to_captcha(vec):
    text = []
    #把概率小于0.5的改为0，标记为错误
    vec[vec < 0.5] = 0
        
    char_pos = vec.nonzero()[0]
    
    for i, ch in enumerate(char_pos):
        text.append(captcha_word[ch % word_class])
    return ''.join(text)

print('Data preprocess start.')
#获取目录下样本列表
image_list = []
for item in os.listdir(train_dir):
    image_list.append(item)
np.random.shuffle(image_list)

# 3代表图片的通道数
X = np.zeros((len(image_list), height, width, 3), dtype = np.uint8)
# 创建数组，储存标签信息
y = np.zeros((len(image_list), word_len * word_class), dtype = np.uint8)

for i,img in enumerate(image_list):
    if i % 10000 == 0:
        print(i)
    img_path = train_dir + "/" + img
    #读取图片
    raw_img = image.load_img(img_path, target_size=(height, width))
    #将图片转为np数组
    X[i] = image.img_to_array(raw_img)
    #将标签转换为数组进行保存
    y[i] = captcha_to_vec(img.split('.')[0])

print('Data preprocess done.')

#保存成pkl文件
file = open('../dataset/captcha_train_data.pkl','wb')
pickle.dump((X,y), file)