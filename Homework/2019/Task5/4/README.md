### 项目功能：判断sql语句是否异常

### 项目内容：
利用Word2vec+LSTM实现了一个Sql注入语句检测器。最终使用flask技术搭建了一个简易的网站用于用户输入
可疑的sql语句，服务器端在预测之后将结果返回到页面中。

### 项目结构
```
|-- data
    |-- sqli.txt 异常数据样本
    |-- normal_examples.txt 正常数据样本
    |-- names.txt 分类名称
|-- model
    |-- lstm 存放好训练好的模型
|-- static
    |-- style.css 网页样式
|-- templates
    |-- _formhelpers.html 帮助界面 
    |-- results.html 返回预测结果界面
    |-- sqlform.html 接收用户输入界面
|-- app.py flask主文件
|-- word2vec.py 处理数据集，生成word2vec矩阵
|-- lstm.py 神经网络模型
|-- detect.py 
|-- utils.py 

```
### 实验环境

- Python 3.6
- Tensorflow ==1.14
- Numpy
- Flask
- sklearn
- wtforms

### 运行

- 如果需要重新训练模型：
    1. 运行word2vec.py将文本数据转换为可以利用的特征向量
    2. 运行lstm.py进行训练并生成模型
- 如果需要直接进行预测：
    1. 运行detect.py 加上你需要预测的语句
    2. 运行app.py 访问相应的网站进行预测

### 实验结果

- 经过10次epoch后，测试集的准确率达到了0.987，召回率达到了0.986

![0.png](https://pic.superbed.cn/item/5dc0128e8e0e2e3ee9f6bd97.jpg)

- 其余各项指标的结果

![3.png](https://pic.superbed.cn/item/5dc012cf8e0e2e3ee9f6c153.jpg)

- 输入框

![4.png](https://pic.superbed.cn/item/5dc013d08e0e2e3ee9f6da16.jpg)

- 当输入正常请求时：

![1.png](https://ae01.alicdn.com/kf/H62c4edd04ec04b5e83b8c7f42f2e12fcu.jpg)

- 当输入了sql注入语句时：

![2.png](https://pic.superbed.cn/item/5dc011108e0e2e3ee9f6a9eb.jpg)

