import os
from PIL import Image
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



def read_train_data():
    """
    读取训练集文件夹下的单字母/数字图像文件
    :return:image_array, image_label:图像list、图像label list
    """
    image_array = []
    image_label = []
    for label in os.listdir(train_data_path):#获取目录下的所有文件
        label_path = train_data_path + '/' + label
        for image_path in os.listdir(label_path):
            image = Image.open(label_path + '/' + image_path)
            image_array.append(image)
            image_label.append(label)
    return image_array, image_label



#feature generated
def feature_transfer(image):
    """
    生成特征矩阵
    计算每副图像的行和、列和，共image_width + image_height个特征
    :param image:图像list
    :return:
    """
    image = image.resize((image_width, image_height)) #标准化图像格式

    feature = []#计算特征
    for x in range(image_width):#计算行特征
        feature_width = 0
        for y in range(image_height):
            if image.getpixel((x, y)) == 0:
                feature_width += 1
        feature.append(feature_width)

    for y in range(image_height): #计算列特征
        feature_height = 0
        for x in range(image_width):
            if image.getpixel((x, y)) == 0:
                feature_height += 1
        feature.append(feature_height)
    return feature


def main():
    image_array, image_label = read_train_data()
    image_feature = []
    for num, image in enumerate(image_array):
        feature = feature_transfer(image)
        image_feature.append(feature)
    return image_feature, image_label


if __name__ == '__main__':
    main()
