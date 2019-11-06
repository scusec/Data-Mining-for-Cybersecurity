# SQL注入攻击检测

## SQL注入
SQL注入，是发生于应用程序与数据库层的安全漏洞。简而言之，是在输入的字符串之中注入SQL指令，在设计不良的程序当中忽略了字符检查，那么这些注入进去的恶意指令就会被数据库服务器误认为是正常的SQL指令而运行，因此遭到破坏或是入侵。

## 文件结构
  ```
  |-- code //相关代码
      |-- SQL.ipynb
  |-- Screen
      |-- Frame.jpg //系统框架图
      |-- test.jpg //测试截图
  |-- README.md
  ```

## 数据集来源
采用现有数据集，
包含正常语句5000条，
SQL语句4956条。

## 实验环境
 - sklearn
 - pandas
 - re

## 特征选取
共计选取了7个特征
 - 语句的长度
 - SQL常用词数量
 - 大写字母频率
 - 数字字符频率
 - 空格频率
 - 标点符号频率
 - 特殊字符频率

## 实现框图

![https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task5/10/Screen/Frame.jpg]()


## 代码基本流程
1. 提取特征
2. 划分训练集合测试集
3. 采用多种机器学习算法训练模型，并对比
4. 对输入的语句进行预测


