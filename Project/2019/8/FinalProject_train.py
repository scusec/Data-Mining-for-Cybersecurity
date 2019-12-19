# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:30:47 2019

@author: Birdman
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
np.warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)#取消科学计数法显示
np.set_printoptions(threshold = np.inf) #全部显示数据不用省略号代替

flags = tf.app.flags
FLAGS = flags.FLAGS



#结构体，服务于load_data()，使之返回特征和类标
class Rdata():
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def make_struct(self, data, target):
        return self.Struct(data, target)


#读取csv文件，并返回数据和标签
def load_data(filename): 
    #清空模型图
    #tf.reset_default_graph()
    sess = tf.InteractiveSession()
    with open(filename,'rt') as raw_data:#打开指定路径下的文件并读取
        readers = pd.read_csv(raw_data, encoding="utf-8", header=0)#将csv文件存入numpy数组中
        data = readers[readers.columns[1:-1]].values
        target_num = readers['target'].values
        target = tf.one_hot(indices = target_num, depth=3, on_value = 1., off_value = 0., axis = 1 , name = "a").eval()#!!
        scaler = preprocessing.StandardScaler()
        data = scaler.fit_transform(data)
        return Rdata(data = data, target = target)

def show_confMat(confusion_mat, classes_name):
    """
    可视化混淆矩阵，保存png格式
    :param confusion_mat: nd-array
    :param classes_name: list,各类别名称
    :return:
    """
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
 
    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()
 
    # 设置文字
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=60)
    plt.yticks(xlocations, classes_name)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix')
 
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.show()
    plt.close()


def train():
    
    #清空模型图
    tf.reset_default_graph()
    
      
    #读入数据
    info_train = load_data(r'.\userinfo.csv')
    X = info_train.data
    Y = info_train.target
    #print(X)
    X, Y = shuffle (X, Y, random_state = 5)
    
    X_train, X_test ,Y_train, Y_test= train_test_split(X, Y,test_size=0.3, random_state = 20, shuffle=True)
    y0 = 0
    y1 = 0
    y2 = 0
    for i in range(len(Y_test)):
        if Y_test[i][0] == 1:
            y0 += 1
        elif Y_test[i][1] == 1:
            y1 += 1
        elif Y_test[i][2] == 1:
            y2 += 1
    print(y0, y1, y2)
    
    #建立神经网络
    x = tf.placeholder(tf.float32, [None, 8], name = 'input')#输入
    W1 = tf.Variable(tf.zeros([8, 3]))#隐藏层!!
    #W2 = tf.Variable(tf.zeros([6, 3]))
    #b1 = tf.Variable(tf.zeros([6]))#!!
    b2 = tf.Variable(tf.zeros([3]))#!!
    
    #f1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    y = tf.nn.softmax(tf.matmul(x, W1) + b2)
    tf.add_to_collection('result', y)
    
    #定义损失函数和优化器
    y_ = tf.placeholder(tf.float32, [None, 3], name = 'output')#!!
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    
    saver=tf.train.Saver(max_to_keep=1)#保存最后一次训练的模型
    
    sess = tf.InteractiveSession()#初始化图
    #训练模型
    tf.initialize_all_variables().run()
    count = 0
    
    count += 1
    cost_list = []
    acc_list = []
    conf_mat = np.zeros([3, 3])
    
    for i in range(2000):
        '''
        Xtr=X[train_index]
        Ytr=Y[train_index]
        Xt=X[test_index]
        Yt=Y[test_index]
        '''
        Xtr = X_train
        Ytr = Y_train
        Xt = X_test
        Yt = Y_test
        batch_xs, batch_ys = Xtr , Ytr
        train_step.run({x: batch_xs, y_: batch_ys})
        cost = sess.run (cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
        cost_list.append(cost)
        # 测试返回准确度
        prediction_value = sess.run(y, feed_dict = {x: Xt})
        premax = np.eye(prediction_value.shape[1])[prediction_value.argmax(1)]
        #print(Yt)
        #print(prediction_value)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = accuracy.eval({x: Xt, y_: Yt}) 
        acc_list.append(acc)
        if i % 100 == 0:
            print ('第 %d 次迭代, 损失值为 %.5f, 分类准确率为 %.5f' % (i + 1, cost, acc))
        for i in range(len(premax)):
            true_i = np.argmax(Yt[i])
            pre_i = np.argmax(premax[i])
            conf_mat[true_i, pre_i] += 1.0
    
    saver.save(sess, './model/weibo_modle')
    print('Model Updated!')
    print('——————————————————————————————————')
    print('小结：')
    print('平均损失值为 %.5f，平均准确率为 %.5f' % (np.mean(cost_list), np.mean(acc_list)))
    n = np.arange(0,len(acc_list),1)
    plt.plot(n,acc_list,color='r',label='Acc')
    plt.plot(n,cost_list,color='b',label='Loss')
    plt.xlabel('Rounds')    #x轴表示
    plt.ylabel('')   #y轴表示
    plt.title("Loss/Acc")      #图标标题表示
    plt.legend()            #每条折线的label显示
    plt.show()               #显示图片
    plt.close()
    
    show_confMat(conf_mat, ['0','1','2'])
    
    
    #conf_mat = np.zeros([3, 3])
    

train()