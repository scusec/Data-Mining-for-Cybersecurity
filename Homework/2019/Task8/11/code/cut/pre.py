from pytesseract import *
from PIL import Image
import os
import shutil

# import pytesseract
# from PIL import Image
# print (pytesseract.image_to_string(Image.open("/Users/dqy/My/captcha/test/1.png"), config="--psm 10"))
img_path = "/Users/dqy/My/captcha/checkcode/"
pic_path = "/Users/dqy/My/captcha/crop_img/"
cate_path = "/Users/dqy/My/captcha/cate/"
#ocr图像识别
def ocr(img):
    try:
        img = Image.open(img)
        rs = pytesseract.image_to_string(img, lang="eng",config="--psm 10")
    except:
        return "none"
    return rs

#使用ocr进行训练集的预分类
def category(originfile, dirs, filename):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    shutil.copyfile(originfile, dirs+filename)

if __name__ == '__main__':
    for fr in os.listdir(pic_path):
        f=pic_path+fr
        if f.rfind(u'.DS_Store')==-1:
            rs = ocr(f)
            category(f,cate_path+"%s/"%rs,fr)