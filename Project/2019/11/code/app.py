from flask import Flask
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField
from wtforms.validators import Length
import time
# from keras.models import load_model
# model = load_model('model.h5')
app = Flask(__name__)

class SQLForm(Form):
    sql = TextAreaField('', validators=[Length(0, 500, message='长度不正确')])

@app.route('/')
def index():
    form = SQLForm(request.form)
    return render_template('index.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    # sqls=[]
    # pred_y=[]
    form = SQLForm(request.form)
    if request.method == 'POST' and form.validate():
        # sql = request.form['sql']
        # sqls.append(sql)
        # y = detect(tovector(sqls),len(sqls))
        # for i in y:
        #     if int(i) == 0:
        #         pred_y.append('normal')
        #     elif int(i) == 1:
        #         pred_y.append('malicious')
        time.sleep(3)
        return render_template('results.html',
                                content="hh",
                                prediction="pred_y",)
    return render_template('index.html', form=form)

if __name__ == '__main__':
   app.run()