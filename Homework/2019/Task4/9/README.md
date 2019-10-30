# 婉姐姐的README

## 文件结构
```
|-- 9
    |-- README.md
    |-- code
    |   |-- bad.txt
    |   |-- big.txt （到数据集给出网址下载）
    |   |-- DGA.txt （到数据集给出网址下载）
    |   |-- gib_model.pki
    |   |-- good.txt
    |   |-- main.ipynb
    |   |-- normal.csv （到数据集给出网址下载）
    |   |-- .ipynb_checkpoints
    |       |-- main-checkpoint.ipynb
    |-- screen
        |-- 决策树模型.png
        |-- 模型验证.png
        |-- 系统框图.png
        |-- 逻辑斯蒂回归模型.png
        |-- 随机森林模型.png
```

## 数据集

### 训练数据

#### 1. 主模型训练数据

DGA.txt

https://data.netlab.360.com/feeds/dga/dga.txt

normal.csv

http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip

#### 2. 马尔科夫链训练数据

big.txt

 https://github.com/exp0se/dga_detector/blob/master/gib/big.txt 

### 测试数据

bad.txt

 https://github.com/exp0se/dga_detector/blob/master/gib/bad.txt 

good.txt

 https://github.com/exp0se/dga_detector/blob/master/gib/good.txt 



## 训练方法

![详见screen目录下的系统框图](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task4/9/screen/%E7%B3%BB%E7%BB%9F%E6%A1%86%E5%9B%BE.png)



## 训练结果

### 逻辑斯蒂回归模型

准确率：86.0%

召回率：87.5%

### 决策模型

准确率：86.0%

召回率：87.54%

### 随机森林模型

准确率：86.0%

召回率：87.54%



## 模型使用

运行ipynb文件，会显示输入框，在输入框输入即可

