import numpy as np
import image_process, image_feature, image_model
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



def main():
    # image_process.main() #处理原始验证码，并存到文件
    # feature, label = image_feature.main() #特征处理

    #特征处理
    image_array, label = image_feature.read_train_data()
    feature = []
    for num, image in enumerate(image_array):
        feature_vec = image_feature.feature_transfer(image)
        # print('label: ',image_label[num])
        # print(feature)
        feature.append(feature_vec)
    print(np.array(feature).shape)
    print(np.array(label).shape)

    #训练模型
    result = image_model.trainModel(feature, label)




if __name__ == '__main__':
    main()


