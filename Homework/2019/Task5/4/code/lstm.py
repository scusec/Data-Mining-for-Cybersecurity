import tensorflow as tf
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from numpy import argmax
from word2vec import *
from sklearn.metrics import confusion_matrix

CONTINUE_TRAIN = False

y_train_dir = './bins/y_train.npy'
x_train_dir = './bins/x_train.npy'
vec_dir = "./bins/word2vec.model"  # word2vec存放位置
classes_voc_dir = './bins/classes_voc.pkl'

random_state=1
print_step=10
TEST_SIZE=0.25
# 超参数
lr = 0.0001
training_iters = 100000 # 迭代次数（不是epoch数） epoch=training_iters/data_len
# training_iters =0
BATCH_SIZE = 128  #训练时批的大小

n_inputs = 100 # 输入维度，等于embedding大小
n_steps = 30  # time steps
n_hidden_units = 60  # 隐藏层层数（不一定等于时序长度）
n_classes = 2
input_keep_prob = 0.6  # dropout层


# 随机种子
tf.set_random_seed(1)



# tf input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# weights
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

batch_size=tf.placeholder(tf.int32,[])

def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    # into hidden
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # basic LSTM Cell.
    cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_hidden_units), input_keep_prob=input_keep_prob)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)
    return results



pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 代价函数 交叉熵
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def train(x_train, y_train, sess):
    data_len = y_train.shape[0]

    saver = tf.train.Saver()
    if (not CONTINUE_TRAIN):
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        # saver.restore(sess, "model/lstm")
        pass
    step = 0
    i = 0
    x_train, y_train = shuffle_data(x_train, y_train)
    while step * BATCH_SIZE < training_iters:
        if ((i + 1) * BATCH_SIZE >= data_len - 1):
            print('******************* An epoch finished *******************')
            # saver.save(sess, "model/lstm")  # 保存模型
            i = 0
            x_train, y_train = shuffle_data(x_train, y_train)  # 每经过一次epoch，洗牌一次
        batch_xs = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_ys = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_xs = np.array(list(batch_xs)).reshape([BATCH_SIZE, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
            batch_size:BATCH_SIZE
        })
        if step % print_step == 0:
            print('train_accuracy: ', sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
                batch_size:BATCH_SIZE
            }))
        step += 1
        i += 1
    # 保存参数
    saver.save(sess, "./model/lstm")  # 保存模型


def predict(x_test, sess, data_num=0):

        # saver = tf.train.Saver()
        # saver.restore(sess, "model/lstm")

    pred_y = np.array(())

    loop=int(data_num / BATCH_SIZE)
    for i in range(loop):
        pred_y = np.append(pred_y,
                           sess.run(tf.argmax(pred, 1),
                                    feed_dict={x: x_test[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE], batch_size: BATCH_SIZE}))
    if(data_num %BATCH_SIZE==0):
        return pred_y
    pred_y=np.append(pred_y,
                        sess.run(tf.argmax(pred, 1),
                        feed_dict={x: x_test[loop * BATCH_SIZE:],
                        batch_size:data_num %BATCH_SIZE}))
    return pred_y



if __name__ == "__main__":
    with open(classes_voc_dir,'rb') as f:
        classes=pickle.load(f)
    data = [i for i in range(len(classes))]
    values = array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)
    

    vx_train=np.load(x_train_dir)
    vy_train=np.load(y_train_dir)
    vy_train=onehot_encoder.transform(vy_train.reshape(-1,1))
    

    x_train,x_test,y_train,y_test=train_test_split(vx_train,vy_train,test_size=TEST_SIZE,random_state=random_state)
    with tf.Session() as sess:

        train(x_train, y_train,sess)  # 训练并保存模型参数，注意y要先进行onehot编码
        pred_y=predict(x_test,sess,y_test.shape[0])

    y_test_real=[]
    for i in range(y_test.shape[0]):
        y_test_real.append(label_encoder.inverse_transform([argmax(y_test[i, :])]))

    print(classes)
    print('test acc:',accuracy_score(y_test_real,pred_y))
    print('test recall:',recall_score(y_test_real,pred_y,average='macro'))
    eva_with_mat(confusion_matrix(y_test_real,pred_y),list(classes.keys()))

