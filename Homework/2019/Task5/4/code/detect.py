from lstm import *

model = Word2Vec.load(vec_dir)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "./model/lstm")


def tovector(payloads):
    payloads_seged=[]
    for payload in payloads:
        tempseg = segment(payload)
        payloads_seged.append(tempseg)
    print(payloads_seged)
    x = []
    tempvx = []
    for payload in payloads_seged:
        for word in payload:
            try:
                tempvx.append(model.wv.get_vector(word))
            except KeyError as e:
                tempvx.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位
        tempvx = np.array(tempvx)
        x.append(tempvx)
        tempvx = []
     # 字符串向量长度填充
    lenth = time_step
    temp_x = []
    for i in range(len(payloads)):
        if (x[i].shape[0] < lenth):
            temp_x.append(np.pad(x[i], ((0, lenth - x[i].shape[0]),
                                        (0, 0)), 'constant', constant_values=0))
        else:
            temp_x.append(x[i][0:lenth])
    temp_x = np.array(temp_x)
    temp_x.reshape(-1, n_steps, n_inputs)
    return temp_x


def detect(x_test, data_num=0):
    pred_y = np.array(())
    loop = int(data_num / BATCH_SIZE)
    for i in range(loop):
        pred_y = np.append(pred_y,
                           sess.run(tf.argmax(pred, 1),
                                    feed_dict={x: x_test[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE],
                                               batch_size: BATCH_SIZE
                                               }))
    if(data_num % BATCH_SIZE == 0):
        return pred_y
    # 最后一个bacth
    pred_y = np.append(pred_y,
                       sess.run(tf.argmax(pred, 1),
                                feed_dict={x: x_test[loop * BATCH_SIZE:],
                                           batch_size: data_num % BATCH_SIZE}))
    return pred_y

if __name__=='__main__':
    res={'normal':0,'malicious':1}
    a=['<script>x=document.createElement(\%22iframe%22);x.src=%22',
    '\'||1']
    pred_y=detect(tovector(a),len(a))
    print(pred_y)
    for i in pred_y:
        if int(i) == 0:
            print('normal')
        elif int(i) == 1:
            print('malicious')

