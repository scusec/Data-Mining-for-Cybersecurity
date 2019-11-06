#! /usr/bin/env python
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField
from wtforms.validators import Length
from lstm import *

model = Word2Vec.load(vec_dir)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "./model/lstm")
app = Flask(__name__)


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

######## Flask
class SQLForm(Form):
    sql = TextAreaField('', validators=[Length(0, 500, message='长度不正确')])

@app.route('/')
def index():
    form = SQLForm(request.form)
    return render_template('sqlform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    sqls=[]
    pred_y=[]
    form = SQLForm(request.form)
    if request.method == 'POST' and form.validate():
        sql = request.form['sql']
        sqls.append(sql)
        y = detect(tovector(sqls),len(sqls))
        for i in y:
            if int(i) == 0:
                pred_y.append('normal')
            elif int(i) == 1:
                pred_y.append('malicious')
        return render_template('results.html',
                                content=sql,
                                prediction=pred_y,)
    return render_template('sqlform.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
