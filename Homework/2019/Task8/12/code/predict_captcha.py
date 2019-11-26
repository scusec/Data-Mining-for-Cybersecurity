import numpy as np
import os

from keras.preprocessing import image
from keras.models import load_model
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

#生成字符索引，同时反向操作一次，方便还原
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
    
#测试及预测方法
def break_captcha(test_dir):
    print('Load images from', test_dir, '.')
    #获取目录下样本列表
    image_list = []
    for item in os.listdir(test_dir):
        image_list.append(item)
    np.random.shuffle(image_list)

    #创建数组，储存图片信息。结构为(50321, 36, 120, 3)，50321代表样本个数，然后是宽度和高度。
    # 3代表图片的通道数，如果对图片进行了灰度处理，可以改为单通道 1
    X = np.zeros((len(image_list), height, width, 3), dtype = np.uint8)
    # 创建数组，储存标签信息
    y = np.zeros((len(image_list), word_len * word_class), dtype = np.uint8)

    model = load_model('../model/captcha__model.h5')

    for i,img in enumerate(image_list):
        img_path = test_dir + "/" + img
        #读取图片
        raw_img = image.load_img(img_path, target_size=(height, width))
        #讲图片转为np数组
        X[i] = image.img_to_array(raw_img)
        #讲标签转换为数组进行保存
        y[i] = captcha_to_vec(img.split('.')[0])
        
        X_test = np.zeros((1, height, width, 3), dtype = np.float32)
        X_test[0] = image.img_to_array(X[i])
        
        result = model.predict(X_test)
        
        vex_test = vec_to_captcha(result[0])
        true_test = vec_to_captcha(y[i])
        
        plt.imshow(X[i])
        plt.show()
        print("true:",true_test,"predict:", vex_test)
    
break_captcha('../test')

