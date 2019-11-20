# webshell检测

## PHP Webshell
PHP Webshell本质上是一段PHP的代码，如果直接用PHP 的源代码分析，会出现很多的噪音，如注释内容、花操作等。将PHP Webshell 的源代码转化成仅含执行语句操作的内容，就会一定程度上过滤掉这些噪音。所以本项目借用VLD扩展将PHP代码转化为PHP opcode，再针对opcode数据类型，采用词袋，词频等方法来进行提取关键特征，最后使用分类的算法进行训练。

## 项目结构
```
.
├── black-list 黑名单文件
├── black_opcodes.txt
├── check.py
├── __pycache__
│   └── utils.cpython-36.pyc
├── requirements.txt
├── save 训练好的缓存文件
│   └── gnb.pkl
├── train.py
├── utils.py
├── utils.pyc
├── white-list 白名单文件
└── white_opcodes.txt
4 directories, 949 files
```

## 实验环境
* Ubuntu 18.04
* Python 3.6
* PHP 5.6
* vld 0.14.0

## 数据集
- 白名单: [https://github.com/WordPress/WordPress](https://github.com/WordPress/WordPress)
- 黑名单: [https://github.com/ysrc/webshell-sample](https://github.com/ysrc/webshell-sample)

## 数据处理
- 提取opcode
用Python 的subprocess 模块来进行执行系统操作，获取其所有输出，并用正则提取opcode，再用空格来连接起来。遍历目标文件夹内的所有的PHP文件并生成opcode，最后生成一个列表写入分别写入black_opcodes.txt和white_opcodes.txt。
```
def load_php_opcode(phpfilename):
try:
    output = subprocess.check_output(['php', '-dvld.active=1', '-dvld.execute=0', phpfilename], stderr=subprocess.STDOUT)
    output = str(output, encoding='utf-8')
    tokens = re.findall(r'\s(\b[A-Z_]+\b)\s', output)
    t = " ".join(tokens)
    return t
except:
    return " "
```
- 打标签
把白名单中的PHP opcode 贴上【0】的标签，把黑名单中的PHP opcode 贴上【1】的标签


## 训练函数
- CountVectorizer
```
cv = CountVectorizer(ngram_range=(3, 3), decode_error="ignore", token_pattern=r'\b\w+\b')
X = cv.fit_transform(X).toarray()
```
把文档集合转化成数值矩阵。
- TfidfTransformer
```
transformer = TfidfTransformer(smooth_idf=False)
X = transformer.fit_transform(X).toarray()
```
把数值矩阵规范化为tf-idf
- train_test_split
分配训练集和测试集
- GaussianNB
```
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
```
采用朴素贝叶斯算法进行训练

## 结果
> Accuracy :0.8936170212765957

## 系统流程图
<img style="width:250px;height:500px" src="https://pic.superbed.cn/item/5dd4e1238e0e2e3ee94c9d93.jpg" />

