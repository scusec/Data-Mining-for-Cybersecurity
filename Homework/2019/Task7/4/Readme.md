### 项目功能：利用 Opcode 检测 PHP 文件是否为 Webshell

### 项目内容
- 利用php-vld拓展来提取PHP文件中的序列码
- 利用word2vec将每个文件的Opcode序列转换为可用的特征向量
- 使用LSTM构建模型对样本进行训练，并将模型进行持久化
- 输入可疑的php文件，返回一个预测结果

### 项目结构
```
|-- data
    |-- 120features_20minwords_10context 存储Opcode转化成的特征向量矩阵
    |-- LSTM_webshell_20.model 存储训练好的LSTM模型
    |-- normal.json 训练集：正常PHP文件的Opcode
    |-- websehll.json 训练集：PHP Webshell文件的Opcode
|-- filter_opcode.py 执行php -dvld.active=1 -dvld.execute=0 <filename>，提取文件的opcode
|-- getData.py 生成训练数据集
|-- train_w2v.py 生成特征向量矩阵
|-- train_LSTM.py 拟合LSTM模型
|-- reload.py 用于输入检测未知的php文件
|-- php_vld.sh 下载php_vld的脚本
|-- ws
    |-- php-webshells 用于输入测试的php文件

```

### 实验环境

- Python 3.6.8
- Keras
- NumPy
- gensim
- 注：由于需要安装php-vld，所以在windows时环境下暂时还没能跑通。本实验采用的是Ubuntu18.04。


### 运行

- 如果需要重新训练模型：
    1. 运行train_w2v.py 生成特征向量
    2. 运行train_LSTM.py 生成模型
- 如果需要直接进行预测
    1. 运行reload.py 输出预测结果


### 实验结果

- 经过12次epoch后，测试集的准确率达到了0.9799，loss为0.0543, mean_absolute_error为0.0334

![1.png](https://pic3.superbed.cn/item/5dd39f0a8e0e2e3ee91ed677.jpg)

- 对一个文件夹中的未知PHP文件进行预测

![2.png](https://pic.superbed.cn/item/5dd3a0d38e0e2e3ee91f0f6e.jpg)
![3.png](https://pic.superbed.cn/item/5dd3a0938e0e2e3ee91f09bf.jpg)
![4.png](https://pic1.superbed.cn/item/5dd39ff18e0e2e3ee91ef672.jpg)

### 框架图

![5.png](https://pic.superbed.cn/item/5dd3a1728e0e2e3ee91f3a6a.png)