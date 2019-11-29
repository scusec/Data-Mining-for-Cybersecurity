import os
import time
import numpy
from PIL import Image
from os.path import join
from itertools import groupby

img_path = "/Users/dqy/My/captcha/checkcode/"
pic_path = "/Users/dqy/My/captcha/crop_img/"

def get_bin_table(threshold=190):
    """
    获取灰度转二值的映射table
    :param threshold:
    :return:
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    return table

def depoint(img):
    """
    去噪
    对于像素值>245的邻域像素，判别为属于背景色，如果一个像素上下左右4各像素值有超过2个像素属于背景色，那么该像素就是噪声
    :param img:
    :return:
    """
    pixdata = img.load()
    w,h = img.size
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] > 250:
                count = count + 1
            if pixdata[x,y+1] > 250:
                count = count + 1
            if pixdata[x-1,y] > 250:
                count = count + 1
            if pixdata[x+1,y] > 250:
                count = count + 1
            if count > 2:
                pixdata[x,y] = 255
    return img

def get_crop_imgs(img):
    """
    按照图片的特点,进行切割,这个要根据具体的验证码来进行工作. # 见本例验证图的结构原理图
    分割图片是传统机器学习来识别验证码的重难点，如果这一步顺利的话，则多位验证码的问题可以转化为1位验证字符的识别问题
    :param img:
    :return:
    """
    child_img_list = []
    for i in range(4):
        x = i * 19  # 见原理图
        y = 0
        child_img = img.crop((x, y, x + 19, img.height))
        child_img_list.append(child_img)
    return child_img_list

def get_clear_bin_image(image):
    """
    获取干净的二值化的图片。
    图像的预处理：
    1. 先转化为灰度
    2. 再二值化
    3. 然后清除噪点
    参考:http://python.jobbole.com/84625/
    :type img:Image
    :return:
    """
    width, height = image.size
    image = image.crop([width/9,height/4,width*8/9,height*4/5])
    imgry = image.convert('L')  # 转化为灰度图

    table = get_bin_table()

    out = depoint(imgry)
    img = out.point(table, '1')  # 变成二值图片:0表示黑色,1表示白色
    return img

def process(image_path):
    image = Image.open(image_path)
    img = get_clear_bin_image(image)
    # img.save(pic_path + "%d"%(int(time.time()*10000))+'.png')
    pics = get_crop_imgs(img)
    for pic in pics:
        pic.save(pic_path + "%d"%(int(time.time()*10000))+'.png')

def process_all(dir):
    # print("1")
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.rfind(u'.DS_Store')==-1:
            # print(os.path.join(root, f))
                process(os.path.join(root, f))

if __name__ == '__main__':
    # print("1")
    process_all(img_path)