# encoding=utf-8

# @Time:2019/2/27 9:31
# @Author:wyx
# @File:lab_randomforest.py
import pickle
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def shuffle_batch(X, y, seq_len, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch, len_batch = X[batch_idx], y[batch_idx], seq_len[batch_idx]
        yield X_batch, y_batch, len_batch


x_file = open('../70/x_rand.pickle', 'rb')
y_file = open('../70/y_rand.pickle', 'rb')
len_file = open('../70/len_rand.pickle', 'rb')

x = pickle.load(x_file)
x = x.astype(np.float32)
y = pickle.load(y_file)
y_data = y.astype(np.int32)
seq_len = pickle.load(len_file)
seq_len = seq_len.astype(np.int32)
print(x.shape)
x = x.reshape(-1, 6)

scaler = MinMaxScaler()
scaled_x = scaler.fit_transform(x)

scaled_x = scaled_x.reshape(-1, 51, 6)

#
n_steps = 51
n_lstm_inputs = 6
n_neurons = 16
n_epochs = 100  # 5*10=50
batch_size = 3200

learning_rate = 0.01
n_layers = 4
n_outputs = 2
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, [None, n_steps, n_lstm_inputs], name="input_data")
    y = tf.placeholder(tf.int32, [None], name="input_label")
    seq_length = tf.placeholder(tf.int32, [None], name="seq_length")

with tf.name_scope("LSTM"):
    lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
                  for layer in range(n_layers)]
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    outputs, tmp_states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=seq_length)

with tf.name_scope("Dense"):
    logits = tf.layers.dense(tmp_states[-1][1], n_outputs, name="Logists")
    softmax = tf.nn.softmax(logits, name="Softmax")

with tf.name_scope("Train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    tf.summary.scalar('loss', loss)

with tf.name_scope("Eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init_l = tf.local_variables_initializer()
init = tf.global_variables_initializer()

saver = tf.train.Saver()

summary_op = tf.summary.merge_all()

sum_acc = []
sum_rec = []
sum_pre = []
sum_f1 = []
count = 0
with tf.Session() as sess:
    init.run()
    init_l.run()
    summary_writer = tf.summary.FileWriter("./logs", graph_def=sess.graph_def)
    for i in range(0, 5000, 500):
        X_test, y_test, len_test = scaled_x[i:i + 500], y_data[i:i + 500], seq_len[i:i + 500]
        X_train, y_train, len_train = scaled_x.tolist()[:i], y_data.tolist()[:i], seq_len.tolist()[:i]
        X_train.extend(scaled_x.tolist()[i + 500:])
        y_train.extend(y_data.tolist()[i + 500:])
        len_train.extend(seq_len.tolist()[i + 500:])
        X_train, y_train, len_train = np.array(X_train), np.array(y_train), np.array(len_train)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        len_train = len_train.astype(np.int32)
        len_test = len_test.astype(np.int32)

        for epoch in range(n_epochs):
            for X_batch, y_batch, len_batch in shuffle_batch(X_train, y_train, len_train, batch_size):
                X_batch = X_batch.reshape((-1, n_steps, n_lstm_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, seq_length: len_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch, seq_length: len_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test, seq_length: len_test})
            log = logits.eval(feed_dict={X: X_test, y: y_test, seq_length: len_test})
            recall_d = []
            for i in log:
                t = np.argmax(i)
                recall_d.append(t)
            y_pred_1 = np.array(recall_d)
            all_pos = (y_test > 0).sum()
            tp = (y_test * y_pred_1 > 0).sum()
            print(epoch, "Last batch accuracy:", acc_batch)
            print("Test Recall:", tp / all_pos, "Test accuracy:", acc_test)
            l = sess.run(softmax, feed_dict={X: X_test, y: y_test, seq_length: len_test})
            summary_str = sess.run(summary_op, feed_dict={X: X_test, y: y_test, seq_length: len_test})
            summary_writer.add_summary(summary_str, count)
            count += 1
            score_y = [x[1] for x in l.tolist()]
            y_pred = [1 if x >= 0.5 else 0 for x in score_y]
            pre = metrics.precision_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
            print("Test Precision:", pre, "Test F1-Score:", f1)
        sum_acc.append(acc_test)
        sum_rec.append(tp / all_pos)
        sum_pre.append(pre)
        sum_f1.append(f1)
    l = sess.run(softmax, feed_dict={X: scaled_x, y: y_data, seq_length: seq_len})
    with open("softmax60.pickle", "wb") as f_out:
        pickle.dump(l, f_out)
    tf.train.Saver().save(sess, "./models/models", write_meta_graph=True)

final_acc = sum(sum_acc) / len(sum_acc)
final_rec = sum(sum_rec) / len(sum_rec)
final_pre = sum(sum_pre) / len(sum_pre)
final_f1 = sum(sum_f1) / len(sum_f1)

print("Test accuracy:", final_acc, "Test Recall:", final_rec, "Test Precision:", final_pre, "Test F1-Score:", final_f1)
