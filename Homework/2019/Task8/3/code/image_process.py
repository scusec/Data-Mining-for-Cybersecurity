from PIL import Image
import random
import os
import time
import random
import image_training
import configparser





#原始路径
path = '/home/zgg/Python-3.7.1/captcha/captcha'
#训练集原始验证码文件存放路径
captcha_path = path + '/data/captcha'
#训练集验证码清理存放路径
captcha__clean_path = path + '/data/captcha_clean'
#训练集存放路径
train_data_path = path + '/data/training_data'
#模型存放路径
model_path = path + '/model/model.model'
#测试集原始验证码文件存放路径
test_data_path = path + '/data/test_data'
#测试结果存放路径
output_path = path + '/result/result.txt'

#识别的验证码个数
image_character_num = 4

#图像粗处理的灰度阈值
threshold_grey = 100

#标准化的图像大小
image_width = 8
image_height = 26




def read_captcha(path):
    """
    读取验证码图片
    :param path: 原始验证码存放路径
    :return: image_array, image_label：存放读取的iamge list和label list
    """
    image_array = []
    image_label = []
    file_list = os.listdir(path)#获取captcha文件
    for file in file_list:
        image = Image.open(path + '/' + file)#打开图片
        file_name = file.split(".")[0]
        image_array.append(image)
        image_label.append(file_name)
        # image.close()
    return image_array, image_label


def image_transfer(image_arry, image_label,captcha_clean_save = False):
    """
    图像粗清理
    将图像转换为灰度图像，将像素值小于某个值的点改成白色
    :param image_arry:
    :param captcha_clean_save:
    :return: image_clean:清理过后的图像list
    """
    image_clean = []
    for i, image in enumerate(image_arry):
        image = image.convert('L') #转换为灰度图像，即RGB通道从3变为1
        im2 = Image.new("L", image.size, 255)

        for y in range(image.size[1]): #遍历所有像素，将灰度超过阈值的像素转变为255（白）
            for x in range(image.size[0]):
                pix = image.getpixel((x, y))
                if int(pix) > threshold_grey:  #灰度阈值
                    im2.putpixel((x, y), 255)
                else:
                    im2.putpixel((x, y), pix)

        if captcha_clean_save: #保存清理过后的iamge到文件
            im2.save(captcha__clean_path + '/' + image_label[i] + '.jpg')
        image_clean.append(im2)
    return image_clean



def get_bin_table(threshold=140):
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



def sum_9_region(img, x, y):
    """
    9邻域框,以当前点为中心的田字框,黑点个数,作为移除一些孤立的点的判断依据
    :param img: Image
    :param x:
    :param y:
    :return:
    """
    cur_pixel = img.getpixel((x, y))  # 当前像素点的值
    width = img.width
    height = img.height

    if cur_pixel == 1:  # 如果当前点为白色区域,则不统计邻域值
        return 0

    if y == 0:  # 第一行
        if x == 0:  # 左上顶点,4邻域
            # 中心点旁边3个点
            sum = cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 4 - sum
        elif x == width - 1:  # 右上顶点
            sum = cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1))

            return 4 - sum
        else:  # 最上非顶点,6邻域
            sum = img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 6 - sum
    elif y == height - 1:  # 最下面一行
        if x == 0:  # 左下顶点
            # 中心点旁边3个点
            sum = cur_pixel \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x, y - 1))
            return 4 - sum
        elif x == width - 1:  # 右下顶点
            sum = cur_pixel \
                  + img.getpixel((x, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y - 1))

            return 4 - sum
        else:  # 最下非顶点,6邻域
            sum = cur_pixel \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x, y - 1)) \
                  + img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x + 1, y - 1))
            return 6 - sum
    else:  # y不在边界
        if x == 0:  # 左边非顶点
            sum = img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))

            return 6 - sum
        elif x == width - 1:  # 右边非顶点
            # print('%s,%s' % (x, y))
            sum = img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1))

            return 6 - sum
        else:  # 具备9领域条件的
            sum = img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1)) \
                  + img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 9 - sum




def remove_noise_pixel(img, noise_point_list):
    """
    根据噪点的位置信息，消除二值图片的黑点噪声
    :type img:Image
    :param img:
    :param noise_point_list:
    :return:
    """
    for item in noise_point_list:
        img.putpixel((item[0], item[1]), 1)


def get_clear_bin_image(image):
    """
    图像细清理
    获取干净的二值化的图片。
    图像的预处理：
    1. 先转化为灰度
    2. 再二值化
    3. 然后清除噪点
    参考:http://python.jobbole.com/84625/
    :type img:Image
    :return:
    """
    imgry = image.convert('L')  # 转化为灰度图

    table = get_bin_table()
    out = imgry.point(table, '1')  # 变成二值图片:0表示黑色,1表示白色

    noise_point_list = []  # 通过算法找出噪声点,第一步比较严格,可能会有些误删除的噪点
    for x in range(out.width):
        for y in range(out.height):
            res_9 = sum_9_region(out, x, y)
            if (0 < res_9 < 3) and out.getpixel((x, y)) == 0:  # 找到孤立点
                pos = (x, y)  #
                noise_point_list.append(pos)
    remove_noise_pixel(out, noise_point_list)
    return out


def image_split(image):
    """
    图像切割
    :param image:单幅图像
    :return:单幅图像被切割后的图像list
    """

    #找出每个字母开始和结束的位置
    inletter = False
    foundletter = False
    start = 0
    end = 0
    letters = []
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            pix = image.getpixel((x, y))
            if pix != True:
                inletter = True
        if foundletter == False and inletter == True:
            foundletter = True
            start = x
        if foundletter == True and inletter == False:
            foundletter = False
            end = x
            letters.append((start, end))
        inletter = False
   

    # 因为切割出来的图像有可能是噪声点
    # 筛选可能切割出来的噪声点
    subtract_array = []
    for each in letters:
        subtract_array.append(each[1]-each[0])
    reSet = sorted(subtract_array, key=lambda x:x, reverse=True)[0:image_character_num]
    letter_chioce = []
    for each in letters:
        if int(each[1] - each[0]) in reSet:
            letter_chioce.append(each)

    #切割图片
    image_split_array = []
    for letter in letter_chioce:
        # (切割的起始横坐标，起始纵坐标，切割的宽度，切割的高度)
        im_split = image.crop((letter[0], 0, letter[1], image.size[1]))
        im_split = im_split.resize((image_width, image_height))
        image_split_array.append(im_split)
    return image_split_array[0:int(image_character_num)]




def image_save(image_array, image_label):
    """
    保存图像到文件
    :param image_array: 切割后的图像list
    :param image_label: 图像的标签
    :return:
    """
    for num, image_meta in enumerate(image_array):
        file_path = captcha__clean_path + image_label[num] + '/'
        file_name = str(int(time.time())) + '_' + str(random.randint(0,100)) + '.gif'
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        image_meta.save(file_path  + file_name, 'gif')
    print("complete save: ",num)





def main():
    image_array, image_label = read_captcha(captcha_path) #读取验证码文件
    image_clean = image_transfer(image_array, image_label, captcha__clean_path) #验证码图像粗清理

    for k, each_image in enumerate(image_clean):
        image_out = get_clear_bin_image(each_image) #验证码图像细清理
        split_result = image_split(image_out) #图像切割
        image_save(split_result, image_label[k]) #保存训练图像




if __name__ == '__main__':
    main()


