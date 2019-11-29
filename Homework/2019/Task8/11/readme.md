#验证码检测

验证码的检测主要有两种方式：

- [分割式](#env)
- [端到端](#env1)

本项目对两种方法都进行了实践和对比,文件结构：

![](https://pic3.superbed.cn/item/5de152bc8e0e2e3ee9224535.png)

#<span id="env">分割式</span>

### 数据集获取

1. 网页爬取：https://www.cndns.com/common/GenerateCheckCode.aspx
2. 标签：OCR预分类，使用pytesseract库，后无法识别的人工分类，每个字符一个文件夹（文件夹命名为字符）

### 系统框架

![](https://ae01.alicdn.com/kf/Hcf333b38fd9e4b0bb2649afa21991bafv.png)

### 特征提取

由于验证码中包含数字和字母，而且同一字符的粗细、形状都不同，所以这里特别考虑了不同的特征提取方法：

根据像素点所形成的矩阵，每一行提取三个特征，分别是第n次0（像素点是黑色的点）连续出现的个数，多余三次不计，少于三次计0

### 训练

使用SVM进行训练，由于字符的去噪没有做到非常的完美，以及字符有形状的变化，准确率暂时只达到0.89，之后会考虑更好的去噪方式再进行实验

## 测试

模型能够正确输出测试的图片，详见train.ipynb

#<span id="env1">端到端</span>

###实验环境

- captcha 0.3
- tensorflow 1.13.1
- numpy 
- tqdm 

###数据集

使用python自带的生成验证码的库captcha，支持图片验证码和语音验证码，我们使用的是它生成图片验证码的功能。设置验证码格式为数字加大写字母。

###数据生成器

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
X 的形状是 (batch_size, height, width, 3)
y 的形状是四个 (batch_size, n_class)

### 构建卷积神经网络

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
特征提取部分使用的是两个卷积，一个池化的结构，重复五个 block，然后将它 Flatten，连接四个分类器，每个分类器是36个神经元，输出36个字符的概率。
###训练模型

使用model.fit_generator，使用相同的生成器生成验证集，使用了 Adam 优化器，学习率设置为1e-3。

使用EarlyStopping方法在loss超过3个epoch没有下降以后，就自动终止训练。
使用ModelCheckpoint方法保存训练过程中最好的模型。
```
callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv',append=True),ModelCheckpoint('cnn_best.h5', save_best_only=True)]
model.compile(loss='categorical_crossentropy',optimizer=Adam(1e-4, amsgrad=True),metrics=['accuracy'])
model.fit_generator(train_data, epochs=20,validation_data=valid_data, workers=4,use_multiprocessing=True,callbacks=callbacks)
```

###结果

<img style="width:250px;height:150px" src="https://pic.superbed.cn/item/5ddfc1f38e0e2e3ee9ee28a4.jpg"/>

```
Epoch 20/20
20/20 [==============================] - 54s 3s/step - loss: 1.0480 - c1_loss: 0.1663 - c2_loss: 0.3222 - c3_loss: 0.3558 - c4_loss: 0.2037 - c1_acc: 0.9512 - c2_acc: 0.9074 - c3_acc: 0.8914 - c4_acc: 0.9371
20/20 [==============================] - 275s 14s/step - loss: 0.8244 - c1_loss: 0.1481 - c2_loss: 0.2636 - c3_loss: 0.2652 - c4_loss: 0.1476 - c1_acc: 0.9500 - c2_acc: 0.9187 - c3_acc: 0.9285 - c4_acc: 0.9484 - val_loss: 1.0480 - val_c1_loss: 0.1663 - val_c2_loss: 0.3222 - val_c3_loss: 0.3558 - val_c4_loss: 0.2037 - val_c1_acc: 0.9512 - val_c2_acc: 0.9074 - val_c3_acc: 0.8914 - val_c4_acc: 0.9371
```

###系统流程图

<img style="width:250px;height:500px" src="https://pic3.superbed.cn/item/5dde2a168e0e2e3ee9b769cc.jpg"/>