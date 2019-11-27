# 验证码检测

## 实验环境
- captcha 0.3
- tensorflow 1.13.1
- numpy 
- tqdm 

## 数据集
使用python自带的生成验证码的库captcha，支持图片验证码和语音验证码，我们使用的是它生成图片验证码的功能。设置验证码格式为数字加大写字母。

## 数据生成器
需要使用 Keras 的 Sequence 类实现数据生成器：
```
class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, n_len=4, width=128, height=64):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)
    
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        for i in range(self.batch_size):
            random_str = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            X[i] = np.array(self.generator.generate_image(random_str)) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y
```
## 构建卷积神经网络
```
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
    for j in range(n_cnn):
        x = Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

x = Flatten()(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
model = Model(inputs=input_tensor, outputs=x)
```
## 模型
模型的大小是10.7MB，总体准确率是 98.26%，基本上可以确定破解了此类验证码。

## 系统流程图
<img style="width:250px;height:500px" src="https://pic3.superbed.cn/item/5dde2a168e0e2e3ee9b769cc.jpg"/>
