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


captcha_chars = ['0','1','2','3','4','5','6','7','8','9']

image_path = 'images/'
test_path = 'test/'
model_path = 'model/'


# 生成随机的验证码图片
def captcha_generator(word_list=captcha_chars, word_listLen=10, image_path=image_path):
    k = 0
    total = 1
    for i in range(4):
        total *= word_listLen

    for i in range(word_listLen):
        for j in range(word_listLen):
            for m in range(word_listLen):
                for n in range(word_listLen):
                    captcha_text = word_list[i] + word_list[j] + word_list[m] + word_list[n]
                    image = ImageCaptcha()
                    image.write(captcha_text, image_path + captcha_text + '.png')
                    k += 1
                    sys.stdout.write("\rCreating images: %d/%d" % (k, total))
                    sys.stdout.flush()


def preprocess_captcha(image_path, TestImgPath):
    fileNameList = []
    for filepath in os.listdir(image_path):
        captcha_image = filepath.split('/')[-1]
        fileNameList.append(captcha_image)

    random.seed(time.time())
    random.shuffle(fileNameList)
    for i in range(100):
        name = fileNameList[i]
        shutil.move(image_path+name, TestImgPath+name)



# 获取图片名，也就是真实的标签
def get_img_name(image_path=image_path):
    fileName = []
    total = 0
    for filePath in os.listdir(image_path):
        captcha_name = filePath.split('/')[-1]
        fileName.append(captcha_name)
        total += 1
    #  print(fileName)
    return fileName, total



def get_label(name):
    label = np.zeros(4 * 10)
    for i, c in enumerate(name):
        idx = i * 10 + ord(c) - ord('0')
        label[idx] = 1

    return label


def connect_name_n_label(fileName, image_path=image_path):
    #  将验证码的图片路径拼接起来
    filepath = os.path.join(image_path, fileName)
    Img = Image.open(filepath)
    #   将验证码转化为灰度图片
    Img = Img.convert("L")
    Img_array = np.array(Img)
    Img_data = Img_array.flatten() / 255
    Img_label = get_label(fileName[0:4])

    return Img_data, Img_label


def next_batch(batchSize=64, trainOrtest='train', step=0):
    batch_data = np.zeros([batchSize, 160 * 60])
    batch_label = np.zeros([batchSize, 4 * 10])
    fileNameList = TRAINING_IMAGE_NAME
    if (trainOrtest == 'validate'):
        fileNameList = VALIDATION_IMAGE_NAME

    totalNumber = len(fileNameList)
    indexStart = step * batchSize

    for i in range(batchSize):
        index = (i + indexStart) % totalNumber
        name = fileNameList[index]
        img_data, img_label = connect_name_n_label(name)
        batch_data[i, :] = img_data
        batch_label[i, :] = img_label

    return batch_data, batch_label


# 定义卷积神经网络
def train_data_with_CNN():
    def weight_variable(shape, name="weight"):
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var

    def bias_variable(shape, name="bias"):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(initial_value=init, name=name)
        return var

    #卷积层
    def conv2d(x, W, name="conv2d"):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

    # 池化层
    def maxPool(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # 1、输入层
    X = tf.placeholder(tf.float32, [None, 160 * 60], name="data_input")
    Y = tf.placeholder(tf.float32, [None, 4 * 10], name="label_input")
    x_input = tf.reshape(X, [-1, 60, 160, 1], name="x-input")
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')

    # 2、卷积一
    W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = maxPool(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # 3、卷积二
    W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
    conv2 = maxPool(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # 4、卷积三
    W_conv3 = weight_variable([5, 5, 64, 64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = maxPool(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # FC
    W_fc1 = weight_variable([20 * 8 * 64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, 20 * 8 * 64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # O
    W_fc2 = weight_variable([1024, 4 * 10], 'W_fc2')
    B_fc2 = bias_variable([4 * 10], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')

    # 最后一层损失函数选用sigmoid
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 预测结果
    predict = tf.reshape(output, [-1, 4, 10], name='predict')
    labels = tf.reshape(Y, [-1, 4, 10], name='labels')
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
            train_data, train_label = next_batch(64, 'train', steps)
            sess.run([optimizer,labels_max_idx], feed_dict={X: train_data, Y: train_label, keep_prob: 0.75})
            if steps % 100 == 0:
                test_data, test_label = next_batch(100, 'validate', steps)
                acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                print("Steps=%d, Accuracy=%.4f" % (steps, acc))
                if acc > 0.95:
                    saver.save(sess, model_path + "captcha.model", global_step=steps)
                    break
            steps += 1



if __name__ == '__main__':
    # ****** 运行 ******  
    # 第一次运行生成图片，之后注释掉即可
    # captcha_generator()
    # preprocess_captcha(image_path, test_path)

    image_filename_list, total = get_img_name(image_path)
    random.seed(time.time())
    # 打乱顺序
    random.shuffle(image_filename_list)
    trainImageNumber = int(total * 0.6)
    # 分成测试集
    TRAINING_IMAGE_NAME = image_filename_list[: trainImageNumber]
    # 和验证集
    VALIDATION_IMAGE_NAME = image_filename_list[trainImageNumber:]
    train_data_with_CNN()
    print('Finished training.')