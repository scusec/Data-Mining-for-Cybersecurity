# XSS检测
## 概述

该项目主要是对XSS攻击进行识别预测，我们采用了几种方式，最简单的一种是采用机器学习SVM模型，其模型准确率、召回率、F1-score均可达到98%，其次采用了深度学习的方式，用了双向LSTM进行训练，最后精确度在测试集上可达100%，模型大小仅451kb。
数据集采用[deep-xss](https://github.com/das-lab/deep-xss)的数据集

## 文档结构

    |-- code
    |-- model
        |-- svm.m //SVM模型
        |-- model-blstm.h5 //双向LSTM模型
    |-- Screen //存放截图

## SVM
使用SVM进行训练，我们选取的特征如下：

-   payload中各种script的数目
-   `java`的数目
-   `iframe`的数目
-   `< > " ' % ( )`的数目

最后模型的准确率98%，召回率98%，如下图：

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task3/5/Screen/svm_score.png)

## 双向LSTM
模型搭建如下图所示：
![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task3/5/Screen/model-bilstm.png)

在tensorboard中的可视化如下图所示：

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task3/5/Screen/tensorboard-bilstm.png)
最后经过3轮训练后准确率在测试集和验证集上可达100%，如下图所示：

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task3/5/Screen/Bilstm.png)

我们也尝试用词嵌入处理数据，利用一维卷积神经网络去做，但是结果不太好，召回率比较低，且模型较大有12.3M，所以这里不附1DCONV的代码及模型。
### How to run
在`jupyter notebook`中运行即可，测试时调用模型即可，封装好的持久化模型在文件夹model中，svm模型测试时利用joblib库调用模型，双向LSTM模型利用keras.models中的load_model加载模型测试即可,如svm可如下测试：
![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task3/5/Screen/svm_test.png)

