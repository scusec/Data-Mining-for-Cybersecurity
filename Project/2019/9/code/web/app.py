from flask import  render_template
from flask import Flask, request
from keras.preprocessing import sequence
import keras
from keras.models import *
import codecs
app = Flask(__name__)
app.config['SECRET_KEY'] = 'xmhcdl'
DGA_type = {
    0:"Alexa（normal)",
    1:"cryptolocker",
    2:"gameover",
    3:"dyre",
    4:"nymaim",
    5:"virut",
    6:"tinba",
    7:"shiotob",
    8:"banjori",
    9:"ramnit",
    10:"ranbyus",
    11:"simda",
    12:"qakbot",
    13:"murofet",
    14:"necurs",
    15:"post",
    16:"pykspa",
    17:"emotet",
    18:"rovnix",
    19:"pykspa_v1"
}
def predict(X_test, batch_size, modelPath, resultPath):
    X_test = sequence.pad_sequences(X_test, maxlen=75)
    my_model = load_model(modelPath)
    y_test = my_model.predict(X_test, batch_size=batch_size).tolist()
@app.route('/', methods=['GET','POST'])
def index():
    max_index = None
    domain_dit = []
    charList = {}
    confFile = codecs.open(filename="./charList.txt", mode='r', encoding='utf-8', errors='ignore')
    lines = confFile.readlines()
    # 字符序列要从1开始,0是填充字符
    i = 1
    for line in lines:
        temp = line.strip('\n').strip('\r').strip(' ')
        if temp != '':
            charList[temp] = i
            i += 1
    if request.method == 'POST':
        domain = request.form.get("domain")
        x_data = []
        for x in domain:
            try:
                x_data.append(charList[x])
            except:
                print('unexpected char' + ' : ' + x)
                x_data.append(0)
        domain_dit.append(x_data)
        domain_ready = np.array(domain_dit)
        domain_ready = sequence.pad_sequences(domain_ready, maxlen=75)
        keras.backend.clear_session()
        my_model = load_model("./models/lsnb.h5")
        result = my_model.predict(domain_ready).tolist()

        for y in result:
            max_index = 0
            max_num = 0.0

            for i in range(len(y)):
                if y[i] > max_num:
                    max_index = i
                    max_num = y[i]
        result = DGA_type[max_index]
        if max_index == 0:
            is_dga = "False"
            return render_template('index.html', result=None, domain=domain, is_dga=is_dga)
        else:
            is_dga = "True"
            return render_template('index.html', result=result, domain=domain, is_dga=is_dga)
    return render_template('index.html')

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
