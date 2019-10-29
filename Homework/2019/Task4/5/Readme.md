

## **基于机器学习与深度学习的DGA域名检测系统**

### **主要内容**

DGA（域名生成算法）是一种利用随机字符来生成C&C域名，从而逃避域名黑名单检测的技术手段。例如，一个由Cryptolocker创建的DGA生成域xeogrhxquuubt.com，如果我们的进程尝试其它建立连接，那么我们的机器就可能感染Cryptolocker勒索病毒。域名黑名单通常用于检测和阻断这些域的连接，但对于不断更新的DGA算法并不奏效。本项目将结合机器学习与深度学习技术，实现对DGA域名的检测。

### 文件结构

    |-- code //相关代码
        |-- BLSTM.ipynb //双向LSTM
        |-- test_blstm.py //测试文件
        |-- Xgboost&MLP.ipynb 
    |-- model
        |-- mlp.m
        |-- model-blstm.h5
    |-- Screen
        |-- Frame.png //系统框架图
        |-- result_BLSTM.png 
        |-- result_MLP.png 
        |-- test_BLSTM //测试截图

### **实验环境及数据集**

环境：python3.6，tensorflow，sklearn

数据集：

- DGA域名：https://data.netlab.360.com/feeds/dga/dga.txt
- 正常域名： https://www.kaggle.com/cheedcheed/top1m



### 程序框架





### **特征选择**

- 域名的长度
- 域名中数字占的比例
- 元音字母所占的比例
- 2-gram的特征提取
- 2，3，4gram的特征提取



### **算法选择**

XGBoost，MLP，双向LSTM(由于网络结构和上次的一样，所以不再展示)

## Result

我们采用了以上三种算法进行训练，最终多层感知机和双向LSTM表现出了比较好的结果，其中多层感知机的模型准确率达到94%，如下图所示：

双向LSTM的模型准确率达到95%，如下图所示：

## How To Run

打开`code`文件夹中的jupyter notebook代码进行运行即可，算法的测试运行test_blstm即可，对于mlp模型的预测，利用joblib加载模型，对文本数据进行2-gram的特征提取然后进行预测即可。

测试样例如下图所示：







