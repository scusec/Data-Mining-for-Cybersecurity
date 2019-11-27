import tensorflow as tf
from datetime import datetime
from util import get_next_batch
from captcha_gen import CAPTCHA_HEIGHT, CAPTCHA_WIDTH, CAPTCHA_LEN, CAPTCHA_LIST


def weight_variable(shape, w_alpha=0.01):
    """
    初始化权值
    :param shape:
    :param w_alpha:
    :return:
   """
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape, b_alpha=0.1):
    """
    初始化偏置项
    :param shape:
    :param b_alpha:
    :return:
    """
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def conv2d(x, w):
    """
    卷基层 ：局部变量线性组合，步长为1，模式‘SAME’代表卷积后图片尺寸不变，即零边距
    :param x:
    :param w:
    :return:
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    池化层：max pooling,取出区域内最大值为代表特征， 2x2 的pool，图片尺寸变为1/2
    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn_graph(x, keep_prob, size, captcha_list=CAPTCHA_LIST, captcha_len=CAPTCHA_LEN):
    """
    三层卷积神经网络
    :param x:   训练集 image x
    :param keep_prob:   神经元利用率
    :param size:        大小 (高,宽)
    :param captcha_list:
    :param captcha_len:
    :return: y_conv
    """
    # 需要将图片reshape为4维向量
    image_height, image_width = size
    x_image = tf.reshape(x, shape=[-1, image_height, image_width, 1])

    # 第一层
    # filter定义为3x3x1， 输出32个特征, 即32个filter
    w_conv1 = weight_variable([3, 3, 1, 32])    # 3*3的采样窗口，32个（通道）卷积核从1个平面抽取特征得到32个特征平面
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)    # rulu激活函数
    h_pool1 = max_pool_2x2(h_conv1)     # 池化
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)      # dropout防止过拟合

    # 第二层
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_drop1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # 第三层
    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_drop2, w_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)

    """
    原始：60*160图片 第一次卷积后 60*160 第一池化后 30*80
    第二次卷积后 30*80 ，第二次池化后 15*40
    第三次卷积后 15*40 ，第三次池化后 7.5*20 = > 向下取整 7*20
    经过上面操作后得到7*20的平面
    """

    # 全连接层
    image_height = int(h_drop3.shape[1])
    image_width = int(h_drop3.shape[2])
    w_fc = weight_variable([image_height*image_width*64, 1024])     # 上一层有64个神经元 全连接层有1024个神经元
    b_fc = bias_variable([1024])
    h_drop3_re = tf.reshape(h_drop3, [-1, image_height*image_width*64])
    h_fc = tf.nn.relu(tf.matmul(h_drop3_re, w_fc) + b_fc)
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)

    # 输出层
    w_out = weight_variable([1024, len(captcha_list)*captcha_len])
    b_out = bias_variable([len(captcha_list)*captcha_len])
    y_conv = tf.matmul(h_drop_fc, w_out) + b_out
    return y_conv


def optimize_graph(y, y_conv):
    """
    优化计算图
    :param y: 正确值
    :param y_conv:  预测值
    :return: optimizer
    """
    # 交叉熵代价函数计算loss 注意logits输入是在函数内部进行sigmod操作
    # sigmod_cross适用于每个类别相互独立但不互斥，如图中可以有字母和数字
    # softmax_cross适用于每个类别独立且排斥的情况，如数字和字母不可以同时出现
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_conv))
    # 最小化loss优化 AdaminOptimizer优化
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    return optimizer


def accuracy_graph(y, y_conv, width=len(CAPTCHA_LIST), height=CAPTCHA_LEN):
    """
    偏差计算图，正确值和预测值，计算准确度
    :param y: 正确值 标签
    :param y_conv:  预测值
    :param width:   验证码预备字符列表长度
    :param height:  验证码的大小，默认为4
    :return:    正确率
    """
    # 这里区分了大小写 实际上验证码一般不区分大小写,有四个值，不同于手写体识别
    # 预测值
    predict = tf.reshape(y_conv, [-1, height, width])   #
    max_predict_idx = tf.argmax(predict, 2)
    # 标签
    label = tf.reshape(y, [-1, height, width])
    max_label_idx = tf.argmax(label, 2)
    correct_p = tf.equal(max_predict_idx, max_label_idx)    # 判断是否相等
    accuracy = tf.reduce_mean(tf.cast(correct_p, tf.float32))
    return accuracy


def train(height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH, y_size=len(CAPTCHA_LIST)*CAPTCHA_LEN):
    """
    cnn训练
    :param height: 验证码高度
    :param width:   验证码宽度
    :param y_size:  验证码预备字符列表长度*验证码长度（默认为4）
    :return:
    """
    # cnn在图像大小是2的倍数时性能最高, 如果图像大小不是2的倍数，可以在图像边缘补无用像素
    # 在图像上补2行，下补3行，左补2行，右补2行
    # np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))

    acc_rate = 0.95     # 预设模型准确率标准
    # 按照图片大小申请占位符
    x = tf.placeholder(tf.float32, [None, height * width])
    y = tf.placeholder(tf.float32, [None, y_size])
    # 防止过拟合 训练时启用 测试时不启用 神经元使用率
    keep_prob = tf.placeholder(tf.float32)
    # cnn模型
    y_conv = cnn_graph(x, keep_prob, (height, width))
    # 优化
    optimizer = optimize_graph(y, y_conv)
    # 计算准确率
    accuracy = accuracy_graph(y, y_conv)
    # 启动会话.开始训练
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())     # 初始化
    step = 0    # 步数
    while 1:
        batch_x, batch_y = get_next_batch(64)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        # 每训练一百次测试一次
        if step % 100 == 0:
            batch_x_test, batch_y_test = get_next_batch(100)
            acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, keep_prob: 1.0})
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
            # 准确率满足要求，保存模型
            if acc > acc_rate:
                model_path = "./model/captcha.model"
                saver.save(sess, model_path, global_step=step)
                acc_rate += 0.01
                if acc_rate > 0.99:     # 准确率达到99%则退出
                    break
        step += 1
    sess.close()


if __name__ == '__main__':
    train()