# 基于机器学习的webshell检测

## 背景介绍

---

### webshell简介

 webshell是一种基于互联网web程序以及web服务器而存在的一种后门形式，主要通过网页脚本程序和服务器容器所支持的后端程序，在web服务器及其中间件中进行运行。 webshell的隐蔽性较强，在上传时可以绕过服务器可能存在的安全检测工具以及避开网络应用防火墙（WAF）的查杀， 在与攻击者进行传输数据包时其行为也不易被检测或侦察  。同时webshell的危害性也比较大，大量的重要网站均存在对webshell防范性低的情况。

### word2vec 简介

 Word2vec，是一群用来产生词向量的相关模型。这些模型为浅而双层的神经网络，用来训练以重新建构语言学之词文本。网络以词表现，并且需猜测相邻位置的输入词 ，在word2vec中词袋模型假设下，词的顺序是不重要的。训练完成之后，Word2Vec模型可用来映射每个词到一个向量，可用来表示词对词之间的关系，该向量为神经网络之隐藏层。

### php opcode 简介

PHP代码执行的顺序是，第一步是词法分析，第二步是语法分析，第三步是转化为opcode，第四步是顺序执行opcode。

## 实验的思路

### 标记数据集

- webshell文件标记为1
- 正常文件标记为零

### 提取特征

- opcode
- 命令执行类函数
- opcode码长度
- 文件的长度
- 文件信息熵

### 模型训练

- LSTM
- Logistic

## 实验环境

---

- python 3.7
-  python gensim库 
- php 7.1
- php VLD 插件

- 实验数据集：链接: https://pan.baidu.com/s/11b-kxpCvAvxpajgZomobHQ      提取码: bsnt 


## 环境部署

---

VLD 是PECL（PHP 扩展和应用仓库）的一个PHP扩展 。可以查看PHP程序的opcode。 

[VLD下载](http://pecl.php.net/package/vld/0.14.0/windows)

[VLD 安装教程]( https://www.cnblogs.com/miao-zp/p/6374311.html )

## 实验结果

- LSTM:   准确率：94.8%     召回率：84.9%

![LSTM](C:\Users\lenovo1\Desktop\Data-Mining-for-Cybersecurity\Task7\6\Screen\LSTM.png)

##  项目改进

- Logistic回归算法部分的实现还存在一些问题有待解决。
- opcode提取的效率较低，耗时过长，还需要改进。

---





