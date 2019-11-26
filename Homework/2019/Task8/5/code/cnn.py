'''
    包括了生成验证码的代码
'''

from captcha.image import ImageCaptcha
import sys
import os
import random
import time
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


CHAR_SET = ['0','1','2','3','4','5','6','7','8','9']

# 字符集的长度
CHAR_SET_LEN = 10
# 验证码的长度，每个验证码由4个数字组成
CAPTCHA_LEN = 4
TRAIN_IMAGE_PERCENT = 0.6
# 验证码图片的存放路径
CAPTCHA_IMAGE_PATH = 'images/'
# 用于模型测试的验证码图片的存放路径，它里面的验证码图片作为测试集
TEST_IMAGE_PATH = 'test/'
# 用于模型测试的验证码图片的个数，从生成的验证码图片中取出来放入测试集中
TEST_IMAGE_NUMBER = 100
MODEL_SAVE_PATH = 'model/'
#    验证码图片的高宽
CAPTCHA_IMAGE_HEIGHT = 60

CAPTCHA_IMAGE_WIDHT =160

# 生成验证码图片，4位的十进制数字可以有10000种验证码
def generate_captcha_image(charSet=CHAR_SET, charSetLen=CHAR_SET_LEN, captchaImgPath=CAPTCHA_IMAGE_PATH):
    k = 0
    total = 1
    for i in range(CAPTCHA_LEN):
        total *= charSetLen

    for i in range(charSetLen):
        for j in range(charSetLen):
            for m in range(charSetLen):
                for n in range(charSetLen):
                    captcha_text = charSet[i] + charSet[j] + charSet[m] + charSet[n]
                    image = ImageCaptcha()
                    image.write(captcha_text, captchaImgPath + captcha_text + '.png')
                    k += 1
                    sys.stdout.write("\rCreating %d/%d" % (k, total))
                    sys.stdout.flush()

def prepare_captcha_image(CaptchaImgPath,TestImgPath):
    fileNameList = []
    for filepath in os.listdir(CaptchaImgPath):
        captcha_image = filepath.split('/')[-1]
        fileNameList.append(captcha_image)

    random.seed(time.time())
    random.shuffle(fileNameList)
    for i in range(TEST_IMAGE_NUMBER):
        name = fileNameList[i]
        shutil.move(CaptchaImgPath+name, TestImgPath+name)


#  获取训练集的图片名
def get_image_file_name(CaptchaImgPath=CAPTCHA_IMAGE_PATH):
    fileName = []
    total = 0
    for filePath in os.listdir(CaptchaImgPath):
        captcha_name = filePath.split('/')[-1]
        fileName.append(captcha_name)
        total += 1
    #  print(fileName)
    return fileName, total


#   get_image_file_name(CAPTCHA_IMAGE_PATH)

# 将验证码转换为训练时用的标签向量，维数是 40
# 例如，如果验证码是 ‘0296’ ，则对应的标签是
# [1 0 0 0 0 0 0 0 0 0
#  0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 1
#  0 0 0 0 0 0 1 0 0 0]
def nameTolabel(name):
    label = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * CHAR_SET_LEN + ord(c) - ord('0')
        label[idx] = 1

    return label


#  获取验证码的文件名以及对应的标签
def get_name_and_label(fileName, CaptchaImgPath=CAPTCHA_IMAGE_PATH):
    #  将验证码的图片路径拼接起来
    #  /Volumes/study/2018大三上课程/大数据分析/作业/train_picture/3324.jpg
    filepath = os.path.join(CaptchaImgPath, fileName)
    Img = Image.open(filepath)
    #   将验证码转化为灰度图片
    Img = Img.convert("L")
    Img_array = np.array(Img)
    Img_data = Img_array.flatten() / 255
    Img_label = nameTolabel(fileName[0:CAPTCHA_LEN])

    return Img_data, Img_label


#  生成一个训练batch
def get_next_batch(batchSize=32, trainOrtest='train', step=0):
    batch_data = np.zeros([batchSize, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT])
    batch_label = np.zeros([batchSize, CAPTCHA_LEN * CHAR_SET_LEN])
    fileNameList = TRAINING_IMAGE_NAME
    if (trainOrtest == 'validate'):
        fileNameList = VALIDATION_IMAGE_NAME

    totalNumber = len(fileNameList)
    indexStart = step * batchSize

    for i in range(batchSize):
        index = (i + indexStart) % totalNumber
        name = fileNameList[index]
        img_data, img_label = get_name_and_label(name)
        batch_data[i, :] = img_data
        batch_label[i, :] = img_label

    return batch_data, batch_label


# 构建卷积神经网络并进行训练
def train_data_with_CNN():
    # 初始化权值
    def weight_variable(shape, name="weight"):
        # 选取位于正态分布均值=0.1附近的随机值
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var

    # 初始化偏置
    def bias_variable(shape, name="bias"):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(initial_value=init, name=name)
        return var

    # 卷积
    # input：指需要做卷积的输入图像
    # filter：相当于CNN中的卷积核
    # strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    # padding：外面加的一圈0
    # use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
    # name：指定该操作的name

    def conv2d(x, W, name="conv2d"):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

    # 池化
    # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

        # 输入层

    X = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT], name="data_input")
    Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN], name="label_input")
    # 数据重定形状函数 tf.reshape(tensor, shape, name=None)
    # tensor：输入数据
    # shape：目标形状
    # name：名称
    x_input = tf.reshape(X, [-1, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDHT, 1], name="x-input")
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')

    # 第一层卷积
    # dropout,防止过拟合
    # tf.nn.relu激活函数
    W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # 第三层卷积
    W_conv3 = weight_variable([5, 5, 64, 64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # 全链接层
    # 每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    W_fc1 = weight_variable([20 * 8 * 64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, 20 * 8 * 64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # 输出层
    W_fc2 = weight_variable([1024, CAPTCHA_LEN * CHAR_SET_LEN], 'W_fc2')
    B_fc2 = bias_variable([CAPTCHA_LEN * CHAR_SET_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='labels')
    # 预测结果
    # 请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))

    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0

        while(1):
            train_data, train_label = get_next_batch(64, 'train', steps)
            op,pre = sess.run([optimizer,labels_max_idx], feed_dict={X: train_data, Y: train_label, keep_prob: 0.75})
            if steps % 100 == 0:
                test_data, test_label = get_next_batch(100, 'validate', steps)
                acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                print("steps=%d, accuracy=%f" % (steps, acc))
                if acc > 0.95:
                    saver.save(sess, MODEL_SAVE_PATH + "crack_captcha.model", global_step=steps)
                    break
            steps += 1


if __name__ == '__main__':
    image_filename_list, total = get_image_file_name(CAPTCHA_IMAGE_PATH)
    random.seed(time.time())
    # 打乱顺序
    random.shuffle(image_filename_list)
    trainImageNumber = int(total * TRAIN_IMAGE_PERCENT)
    # 分成测试集
    TRAINING_IMAGE_NAME = image_filename_list[: trainImageNumber]
    # 和验证集
    VALIDATION_IMAGE_NAME = image_filename_list[trainImageNumber:]
    train_data_with_CNN()
    print('Training finished')